from collections import deque
from functools import partial
import threading
import traceback
from copy import deepcopy

from core.service_client import ServiceClient, ServiceClientAsync

import rclpy
import numpy as np
import yaml

from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import String

from core.utils import actuation_dict_to_msg
from core_interfaces.srv import GetNodeFromLTM

from cognitive_node_interfaces.msg import Episode as EpisodeMsg
from cognitive_node_interfaces.msg import SuccessRate
from cognitive_node_interfaces.msg import Perception
from cognitive_node_interfaces.msg import Action as ActionMsg
from cognitive_node_interfaces.srv import Execute, Predict, PredictUtility
from cognitive_node_interfaces.msg import ObjectLayout, Actuation, ObjectParameters

from cognitive_processes_interfaces.msg import RewardList

from cognitive_nodes.drive import Drive
from cognitive_nodes.goal import Goal
from cognitive_nodes.policy import Policy, PolicyBlocking
from cognitive_processes.cognitive_process import CognitiveProcess
from cognitive_nodes.episode import Episode, Action as EpisodeAction


class DriveDreaming(Drive):
    def __init__(
        self,
        name="drive_dreaming",
        class_name="cognitive_nodes.drive.Drive",
        ltm_service_name="ltm_0/get_node",
        prediction_error_threshold=0.02,
        discovery_period=5.0,
        candidate_names=None,
        **params
    ):
        super().__init__(name, class_name, **params)

        self.prediction_error_threshold = prediction_error_threshold
        self.discovery_period = discovery_period
        self.candidate_names = candidate_names or []

        self.known_world_models = set()
        self.pending_ltm_queries = set()
        self.world_model_subscribers = {}
        self.world_model_errors = {}
        self.selected_world_model = None
        self.full_ltm_query_pending = False

        self.ltm_client = self.create_client(GetNodeFromLTM, ltm_service_name)

        self.get_logger().info(
            f"DriveDreaming init: threshold={self.prediction_error_threshold}, "
            f"discovery_period={self.discovery_period}, candidate_names={self.candidate_names}, "
            f"ltm_service_name={ltm_service_name}"
        )

        self.discovery_timer = self.create_timer(
            self.discovery_period,
            self.refresh_world_models_from_ltm
        )

    def refresh_world_models_from_ltm(self):
        self.get_logger().debug(
            f"Refreshing world models from LTM... candidate_names={self.candidate_names}, "
            f"known={list(self.known_world_models)}, pending={list(self.pending_ltm_queries)}, "
            f"full_ltm_query_pending={self.full_ltm_query_pending}"
        )

        if not self.ltm_client.service_is_ready():
            self.get_logger().info("LTM service not ready yet")
            return

        if self.candidate_names:
            for node_name in self.candidate_names:
                if node_name in self.known_world_models:
                    self.get_logger().info(f"{node_name} already known")
                    continue
                if node_name in self.pending_ltm_queries:
                    self.get_logger().info(f"{node_name} query already pending")
                    continue

                self.get_logger().info(f"Querying LTM for explicit candidate {node_name}")
                req = GetNodeFromLTM.Request()
                req.name = node_name

                future = self.ltm_client.call_async(req)
                self.pending_ltm_queries.add(node_name)
                future.add_done_callback(
                    partial(self.handle_ltm_response, queried_name=node_name)
                )
            return

        if self.full_ltm_query_pending:
            self.get_logger().info("Full LTM dump query already pending")
            return

        self.get_logger().debug("candidate_names is empty; autodiscovering world models from full LTM dump")
        req = GetNodeFromLTM.Request()
        req.name = ""
        future = self.ltm_client.call_async(req)
        self.full_ltm_query_pending = True
        future.add_done_callback(self.handle_full_ltm_response)

    def handle_ltm_response(self, future, queried_name):
        self.pending_ltm_queries.discard(queried_name)

        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f"LTM query failed for {queried_name}: {e}")
            return

        raw_data = response.data
        self.get_logger().info(
            f"LTM raw response for {queried_name}: {str(raw_data)[:300]}"
        )

        try:
            node_data = yaml.safe_load(raw_data)
        except Exception as e:
            self.get_logger().error(f"YAML parse failed for {queried_name}: {e}")
            return

        if isinstance(node_data, dict):
            self.get_logger().info(
                f"LTM parsed keys for {queried_name}: {list(node_data.keys())}"
            )
        else:
            self.get_logger().info(
                f"LTM parsed type for {queried_name}: {type(node_data)}"
            )

        is_wm = self.is_world_model(node_data)
        self.get_logger().info(f"is_world_model({queried_name}) -> {is_wm}")

        if is_wm:
            self.get_logger().info(f"{queried_name} identified as WorldModel")
            self.known_world_models.add(queried_name)
            self.ensure_subscription(queried_name)
        else:
            self.get_logger().info(f"{queried_name} is NOT a WorldModel")

    def handle_full_ltm_response(self, future):
        self.full_ltm_query_pending = False

        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f"Full LTM dump query failed: {e}")
            return

        raw_data = response.data
        self.get_logger().debug(f"LTM full dump received: {str(raw_data)[:500]}")

        try:
            ltm_data = yaml.safe_load(raw_data)
        except Exception as e:
            self.get_logger().error(f"YAML parse failed for full LTM dump: {e}")
            return

        if isinstance(ltm_data, dict):
            self.get_logger().debug(f"Full LTM top-level keys: {list(ltm_data.keys())}")
        else:
            self.get_logger().debug(f"Full LTM parsed type: {type(ltm_data)}")

        discovered = self.extract_world_model_names(ltm_data)
        self.get_logger().debug(f"Autodiscovered world models: {discovered}")

        if not discovered:
            self.get_logger().debug("No world models discovered in full LTM dump")
            return

        for world_model_name in discovered:
            if world_model_name not in self.known_world_models:
                self.get_logger().info(f"Registering autodiscovered world model {world_model_name}")
            self.known_world_models.add(world_model_name)
            self.ensure_subscription(world_model_name)

    def extract_world_model_names(self, data):
        discovered = set()

        def visit(obj, path="root"):
            if isinstance(obj, dict):
                node_name = obj.get("name") or obj.get("node_name")
                node_type = obj.get("nodetype") or obj.get("node_type")
                class_name = obj.get("class_name") or obj.get("defaultclass")

                if node_name and node_type == "WorldModel":
                    self.get_logger().info(
                        f"Discovered WorldModel by node_type at {path}: {node_name}"
                    )
                    discovered.add(str(node_name))

                if node_name and class_name and "WorldModel" in str(class_name):
                    self.get_logger().info(
                        f"Discovered WorldModel by class_name at {path}: {node_name}"
                    )
                    discovered.add(str(node_name))

                for key, value in obj.items():
                    if key == "WorldModel" and isinstance(value, dict):
                        self.get_logger().debug(
                            f"Discovered WorldModel section at {path}. Keys: {list(value.keys())}"
                        )
                        for wm_name in value.keys():
                            discovered.add(str(wm_name))
                    visit(value, f"{path}.{key}")

            elif isinstance(obj, list):
                for index, item in enumerate(obj):
                    if isinstance(item, dict):
                        item_name = item.get("name") or item.get("node_name")
                        item_type = item.get("nodetype") or item.get("node_type")
                        item_class = item.get("class_name") or item.get("defaultclass")

                        if item_name and item_type == "WorldModel":
                            self.get_logger().info(
                                f"Discovered WorldModel list item by node_type at {path}[{index}]: {item_name}"
                            )
                            discovered.add(str(item_name))

                        if item_name and item_class and "WorldModel" in str(item_class):
                            self.get_logger().info(
                                f"Discovered WorldModel list item by class_name at {path}[{index}]: {item_name}"
                            )
                            discovered.add(str(item_name))
                    visit(item, f"{path}[{index}]")

        visit(data)
        return sorted(discovered)

    def is_world_model(self, node_data):
        if isinstance(node_data, dict):
            node_type = node_data.get("nodetype") or node_data.get("node_type")
            if node_type == "WorldModel":
                return True
            class_name = node_data.get("class_name") or node_data.get("defaultclass")
            if class_name and "WorldModel" in str(class_name):
                return True
            if "WorldModel" in node_data and isinstance(node_data["WorldModel"], (dict, list)):
                return True
            return any("WorldModel" in str(v) for v in node_data.values())
        if isinstance(node_data, list):
            return any(self.is_world_model(v) for v in node_data)
        return "WorldModel" in str(node_data)

    def ensure_subscription(self, world_model_name):
        topic_name = f"/world_model/{world_model_name}/prediction_error"

        if topic_name in self.world_model_subscribers:
            self.get_logger().info(f"Already subscribed to {topic_name}")
            return

        self.get_logger().info(f"Creating subscription to {topic_name}")

        sub = self.create_subscription(
            SuccessRate,
            topic_name,
            lambda msg, wm=world_model_name: self.prediction_error_callback(msg, wm),
            10
        )
        self.world_model_subscribers[topic_name] = sub
        self.get_logger().info(
            f"Subscribed to {topic_name}. Current subscribers: {list(self.world_model_subscribers.keys())}"
        )

    def prediction_error_callback(self, msg, world_model_name):
        self.get_logger().info(
            f"Prediction error callback: world_model={world_model_name}, "
            f"msg.node_name={msg.node_name}, msg.node_type={msg.node_type}, success_rate={msg.success_rate}"
        )
        self.world_model_errors[world_model_name] = msg.success_rate
        self.update_selected_world_model()

    def update_selected_world_model(self):
        candidates = {
            name: err
            for name, err in self.world_model_errors.items()
            if err > self.prediction_error_threshold
        }
        self.selected_world_model = max(candidates, key=candidates.get) if candidates else None

        self.get_logger().info(
            f"Selected world model={self.selected_world_model}, errors={self.world_model_errors}, "
            f"threshold={self.prediction_error_threshold}, candidates_above_threshold={candidates}"
        )

    def evaluate(self, perception=None):
        self.evaluation.evaluation = 1.0 if self.selected_world_model else 0.0
        self.evaluation.timestamp = self.get_clock().now().to_msg()
        self.get_logger().debug(
            f"DriveDreaming evaluate -> {self.evaluation.evaluation} "
            f"(selected_world_model={self.selected_world_model})"
        )
        return self.evaluation


class Dreaming(CognitiveProcess):
    def __init__(
        self,
        name,
        node,
        iterations=50,
        trials=1,
        LTM_id="",
        reward_threshold=0.1,
        **params,
    ):
        super().__init__(name, iterations, trials, LTM_id, **params)
        self.node = node
        self.reward_threshold = reward_threshold
        self.current_reward = 0.0
        self.selected_world_model = None
        self.initial_perception = None

        self.start_flag = threading.Event()
        self.finished_flag = threading.Event()

        self.summary_episode = Episode()
        self.summary_episode.parent_policy = self.node.name

        self.set_attributes_from_params(params)
        self.setup()
        self.start_threading()

    def setup(self):
        super().setup()

    def update_activations(self):
        self.get_logger().info("Updating activations...")
        self.semaphore.acquire()
        self.activation_time = self.get_clock().now().nanoseconds
        for node in self.activation_inputs:
            self.activation_inputs[node]["flag"].clear()

        for node in self.activation_inputs:
            if node == self.node.name:
                self.activation_inputs[node]["flag"].set()
            self.get_logger().debug(f"DEBUG: Waiting for activation: {node}")
            self.activation_inputs[node]["flag"].wait()
            self.activation_inputs[node]["flag"].clear()
        self.semaphore.release()
        self.get_logger().debug("DEBUG - LTM CACHE:" + str(self.LTM_cache))

    def get_linked_goals(self):
        cnodes = [
            neighbor["name"]
            for neighbor in self.node.neighbors
            if neighbor["node_type"] == "CNode"
        ]
        cnodes_neighbors = []
        for cnode in cnodes:
            if cnode in self.LTM_cache.get("CNode", {}):
                cnodes_neighbors.extend(self.LTM_cache["CNode"][cnode]["neighbors"])
        linked_goals = [
            neighbor["name"]
            for neighbor in cnodes_neighbors
            if neighbor["node_type"] == "Goal"
        ]
        self.get_logger().info(f"Linked goals: {linked_goals}")
        return linked_goals or self.get_goals(self.LTM_cache)

    def check_completion(self):
        rewards = [
            self.current_episode.reward_list.get(goal, 0.0)
            for goal in self.active_goals
            if goal in self.current_episode.reward_list
        ]
        self.current_reward = max(rewards) if rewards else 0.0
        achieved = any(r > self.reward_threshold for r in rewards)
        self.get_logger().info(
            f"Dreaming {self.get_name()}: completion check — "
            f"rewards={rewards} achieved={achieved}"
        )
        return achieved

    def sample_random_perception_t0(self):
        perception = deepcopy(self.current_episode.perception)
        if perception is None:
            perception = {}
        if isinstance(perception, dict):
            for key, value in perception.items():
                if isinstance(value, list):
                    perception[key] = [0.0 for _ in value]
                elif isinstance(value, (int, float)):
                    perception[key] = 0.0
                elif isinstance(value, bool):
                    perception[key] = False
        return perception

    def predict_next_perception(self, old_perception, action):
        response = self.node.world_model_client.send_request(
            old_perception=old_perception,
            action=action.actuation,
        )
        predicted = getattr(response, "perception", None)
        if predicted is None:
            predicted = getattr(response, "predicted_perception", None)
        if predicted is None:
            raise RuntimeError("World model response does not contain a prediction")
        return predicted

    def evaluate_utility(self, old_perception, new_perception):
        response = self.node.utility_model_client.send_request(
            old_perception=old_perception,
            new_perception=new_perception,
        )
        reward = getattr(response, "reward", None)
        if reward is None:
            reward = getattr(response, "utility", None)
        if reward is None:
            raise RuntimeError("Utility model response does not contain reward")
        return float(reward)

    def select_dream_action(self, perception):
        return self.node.select_action_with_sac(perception)

    def dreaming_cycle(self):
        self.start_flag.wait()
        self.get_logger().info(f"Dreaming cycle started for {self.get_name()}")

        self.current_episode.perception = self.initial_perception or self.sample_random_perception_t0()
        self.summary_episode.old_perception = self.current_episode.perception

        self.update_activations()
        self.summary_episode.old_ltm_state = deepcopy(self.LTM_cache)
        self.active_goals = self.get_linked_goals()

        achieved = False
        step = 0

        while step < self.iterations and not self.stop and not achieved:
            if self.paused:
                step += 1
                continue

            self.get_logger().info(f"*** DREAMING STEP: {step + 1}/{self.iterations} ***")

            old_perception = deepcopy(self.current_episode.perception)
            actuation = self.select_dream_action(old_perception)
            self.current_episode.action = EpisodeAction(actuation=actuation, policy_id=0)
            self.current_episode.parent_policy = self.node.name

            predicted_t1 = self.predict_next_perception(
                old_perception,
                self.current_episode.action
            )

            self.current_episode.old_perception = old_perception
            self.current_episode.perception = predicted_t1

            reward = self.evaluate_utility(
                self.current_episode.old_perception,
                self.current_episode.perception,
            )
            self.current_episode.reward_list = {"utility_model": reward}

            self.publish_episode()
            achieved = reward > self.reward_threshold
            self.current_reward = reward

            self.get_logger().info(
                f"Dreaming {self.get_name()}: step {step + 1} — "
                f"reward={self.current_reward:.3f} achieved={achieved}"
            )
            step += 1

        self.summary_episode.perception = self.current_episode.perception
        self.summary_episode.reward_list = self.current_episode.reward_list

        self.get_logger().info(
            f"Dreaming cycle finished for {self.get_name()} "
            f"(steps={step}, achieved={achieved})"
        )

        self.finished_flag.set()
        self.start_flag.clear()

    def run(self):
        self.current_episode.perception = self.read_perceptions()
        while True:
            try:
                self.dreaming_cycle()
            except Exception as exc:
                self.get_logger().error(
                    f"Exception in dreaming cycle: {exc}\n{traceback.format_exc()}"
                )
                self.finished_flag.set()
                self.start_flag.clear()
                break


class PolicyDreaming(Policy):
    def __init__(self, name='policy_dreaming', dream_process_class=None, dreaming_drive=None, **params):
        super().__init__(name, **params)
        self.dreaming = None
        self.dreaming_drive = dreaming_drive
        self._sac_lock = threading.Lock()

        if dream_process_class is not None:
            self.dreaming = dream_process_class(
                name=f"{self.name}_process",
                node=self,
                **params
            )

        self.get_logger().info("PolicyDreaming initialized")

    def _perception_dict_to_obs(self, perception):
        obs = []
        if isinstance(perception, dict):
            for value in perception.values():
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            obs.extend(float(v) for v in item.values())
                        else:
                            obs.append(float(item))
                elif isinstance(value, dict):
                    obs.extend(float(v) for v in value.values())
                else:
                    obs.append(float(value))
        return np.asarray(obs, dtype=np.float32)

    def _action_vec_to_dict(self, action_vec):
        return {
            "dream_action": [
                {"value": float(v)} for v in np.atleast_1d(action_vec)
            ]
        }

    def select_action_with_sac(self, perception):
        if not hasattr(self, "_sac") or self._sac is None:
            raise RuntimeError("SAC model is not available in PolicyDreaming")

        obs = self._perception_dict_to_obs(perception)
        with self._sac_lock:
            action_vec, _ = self._sac.predict(obs, deterministic=False)

        self.get_logger().info(
            f"SAC dream action: obs_dim={len(obs)}, action={np.array(action_vec).tolist()}"
        )
        return self._action_vec_to_dict(action_vec)

    def get_selected_world_model(self):
        if self.dreaming_drive is None:
            return None
        return getattr(self.dreaming_drive, "selected_world_model", None)

    async def execute_callback(self, request, response):
        if self.dreaming is None:
            raise RuntimeError("Dreaming process is not configured")

        selected_world_model = self.get_selected_world_model()
        if selected_world_model is None:
            self.get_logger().warning("No selected world model available")
            response.policy = self.name
            return response

        self.get_logger().info(
            f"Executing dreaming policy with world model: {selected_world_model}"
        )

        self.dreaming.selected_world_model = selected_world_model
        self.dreaming.initial_perception = request.perception

        self.dreaming.finished_flag.clear()
        self.dreaming.start_flag.set()
        self.dreaming.finished_flag.wait()

        summary = self.dreaming.summary_episode

        response.policy = self.name
        if hasattr(response, "action") and hasattr(summary, "action"):
            response.action = summary.action.actuation
        if hasattr(response, "perception") and hasattr(summary, "perception"):
            response.perception = summary.perception

        self.get_logger().info(
            f"Dreaming policy finished. Final reward_list={summary.reward_list}"
        )
        return response


class PolicyQueue:
    def __init__(self):
        self.queue = deque()

    def select_policy(self):
        policy = self.front()
        self.queue.rotate(1)
        return policy

    def shuffle(self, rng: np.random.Generator):
        rng.shuffle(self.queue)

    def find_differences(self, items):
        new = [x for x in items if x not in self.queue]
        missing = [x for x in self.queue if x not in items]
        return new, missing

    def merge(self, items):
        new, missing = self.find_differences(items)
        for item in new:
            self.enqueue(item)
        for item in missing:
            self.remove(item)
        if not new and not missing:
            return False
        return True

    def enqueue(self, item):
        return self.queue.appendleft(item)

    def dequeue(self):
        return self.queue.pop()

    def remove(self, item):
        if item in self.queue:
            self.queue.remove(item)
            return True
        return False

    def isEmpty(self):
        return len(self.queue) == 0

    def front(self):
        return self.queue[-1]

    def rear(self):
        return self.queue[0]

    def exists(self, item):
        return item in self.queue

    def __len__(self):
        return len(self.queue)


class PolicyDreamingDummy(Policy):
    def __init__(self, name='policy_dreaming_dummy', **params):
        super().__init__(name, **params)

        self.declare_parameter("world_model_name", "WorldModel_0")
        self.declare_parameter("perception_size", 11)
        self.declare_parameter("action_size", 4)
        self.declare_parameter("utility_model_name", "UtilityModel_0")

        self.utility_model_name = self.get_parameter(
            "utility_model_name"
        ).get_parameter_value().string_value

        self.utility_model_predict_service_name = self._build_utility_model_predict_service_name(
            self.utility_model_name
        )
        self.utility_model_client = None

        self.world_model_name = self.get_parameter(
            "world_model_name"
        ).get_parameter_value().string_value

        self.perception_size = self.get_parameter(
            "perception_size"
        ).get_parameter_value().integer_value

        self.action_size = self.get_parameter(
            "action_size"
        ).get_parameter_value().integer_value

        self.world_model_predict_service_name = self._build_world_model_predict_service_name(
            self.world_model_name
        )
        self.world_model_predict_client = None

        self.add_on_set_parameters_callback(self._on_parameters_changed)

        self.manual_world_model_predict_srv = self.create_service(
            Predict,
            f"/{self.name}/manual_world_model_predict",
            self.manual_world_model_predict_trigger_callback
        )

        self.get_logger().info(
            f"PolicyDreamingDummy initialized. "
            f"world model service={self.world_model_predict_service_name}, "
            f"utility model service={self.utility_model_predict_service_name}, "
            f"manual service=/{self.name}/manual_world_model_predict, "
            f"perception_size={self.perception_size}, action_size={self.action_size}"
        )

    def _build_world_model_predict_service_name(self, world_model_name: str) -> str:
        return f"/world_model/{world_model_name}/predict"

    def _build_utility_model_predict_service_name(self, utility_model_name: str) -> str:
        return f"/utility_model/{utility_model_name}/predict"

    def _ensure_world_model_predict_client(self):
        if self.world_model_predict_client is not None:
            return

        tmp_client = self.create_client(Predict, self.world_model_predict_service_name)
        if not tmp_client.wait_for_service(timeout_sec=2.0):
            raise RuntimeError(
                f"{self.world_model_predict_service_name} not available yet"
            )

        self.world_model_predict_client = ServiceClient(
            Predict,
            self.world_model_predict_service_name
        )

    def _ensure_utility_model_client(self):
        if self.utility_model_client is not None:
            return

        tmp_client = self.create_client(
            PredictUtility,
            self.utility_model_predict_service_name
        )
        if not tmp_client.wait_for_service(timeout_sec=2.0):
            raise RuntimeError(
                f"{self.utility_model_predict_service_name} not available yet"
            )

        self.utility_model_client = ServiceClient(
            PredictUtility,
            self.utility_model_predict_service_name
        )

    def _recreate_world_model_predict_client(self):
        self.world_model_predict_service_name = self._build_world_model_predict_service_name(
            self.world_model_name
        )
        self.world_model_predict_client = None
        self.get_logger().info(
            f"World model predict client reset to {self.world_model_predict_service_name}"
        )

    def _recreate_utility_model_client(self):
        self.utility_model_predict_service_name = self._build_utility_model_predict_service_name(
            self.utility_model_name
        )
        self.utility_model_client = None
        self.get_logger().info(
            f"Utility model predict client reset to {self.utility_model_predict_service_name}"
        )

    def _on_parameters_changed(self, params):
        result = SetParametersResult(successful=True)

        for param in params:
            if param.name == "world_model_name":
                if not param.value or not str(param.value).strip():
                    result.successful = False
                    result.reason = "world_model_name cannot be empty"
                    return result
                self.world_model_name = str(param.value).strip()
                self._recreate_world_model_predict_client()

            elif param.name == "utility_model_name":
                if not param.value or not str(param.value).strip():
                    result.successful = False
                    result.reason = "utility_model_name cannot be empty"
                    return result
                self.utility_model_name = str(param.value).strip()
                self._recreate_utility_model_client()

            elif param.name == "perception_size":
                if int(param.value) <= 0:
                    result.successful = False
                    result.reason = "perception_size must be > 0"
                    return result
                self.perception_size = int(param.value)

            elif param.name == "action_size":
                if int(param.value) <= 0:
                    result.successful = False
                    result.reason = "action_size must be > 0"
                    return result
                self.action_size = int(param.value)

        return result

    def _make_layout(self):
        layout = ObjectLayout()
        layout.dim = []
        layout.data_offset = 0
        return layout

    def _make_perception(self, size: int, fill_value: float = 0.0) -> Perception:
        p = Perception()
        p.layout = self._make_layout()
        p.data = [float(fill_value)] * size
        p.is_valid = [True] * size
        return p

    def _make_action(self, size: int, fill_value: float = 0.0) -> ActionMsg:
        act = ActionMsg()
        act.actuation = Actuation()
        act.actuation.layout = self._make_layout()
        act.actuation.data = [float(fill_value)] * size
        act.actuation.is_valid = [True] * size
        act.policy_id = 0
        return act

    def _make_dummy_episode(self) -> EpisodeMsg:
        ep = EpisodeMsg()
        ep.old_perception = self._make_perception(self.perception_size, 0.0)
        ep.parent_policy = self.name
        ep.action = self._make_action(self.action_size, 0.0)

        ep.perception = self._make_perception(self.perception_size, 0.0)

        rl = RewardList()
        rl.goals = []
        rl.rewards = []
        ep.reward_list = rl

        ep.timestamp = self.get_clock().now().to_msg()
        return ep

    def manual_world_model_predict_trigger_callback(self, request, response):
        try:
            self._ensure_world_model_predict_client()
            self._ensure_utility_model_client()
        except Exception as exc:
            self.get_logger().error(f"[proxy] failed to create clients: {exc}")
            response.output_episodes = []
            response.valid = False
            return response

        self.get_logger().info(
            f"Manual world model predict trigger called. "
            f"Using world model service {self.world_model_predict_service_name} "
            f"and utility model service {self.utility_model_predict_service_name}"
        )

        input_episode = EpisodeMsg()

        old_p = Perception()
        old_p.layout = ObjectLayout()
        old_p.layout.dim = [
            ObjectParameters(object='ball_angle0', labels=['data'], size=8, stride=8, size_stride_units='bytes'),
            ObjectParameters(object='box_angle0', labels=['data'], size=8, stride=8, size_stride_units='bytes'),
            ObjectParameters(object='dist_ball_box0', labels=['distance', 'angle_cos', 'angle_sin'], size=24, stride=8, size_stride_units='bytes'),
            ObjectParameters(object='dist_left_arm_ball0', labels=['distance', 'angle_cos', 'angle_sin'], size=24, stride=8, size_stride_units='bytes'),
            ObjectParameters(object='dist_right_arm_ball0', labels=['distance', 'angle_cos', 'angle_sin'], size=24, stride=8, size_stride_units='bytes'),
        ]
        old_p.layout.data_offset = 0
        old_p.data = [
            0.13624948354867789,
            0.5608655958909254,
            0.5014107877258939,
            0.8669652112813435,
            0.16038767144104005,
            0.3145382182829519,
            0.9983897815549261,
            0.45990479278476004,
            0.7099251685547043,
            0.7390485081174856,
            0.0608464838273472,
        ]
        old_p.is_valid = [True] * len(old_p.data)
        input_episode.old_perception = old_p

        input_episode.parent_policy = self.name

        act = ActionMsg()
        act.actuation = Actuation()
        act.actuation.layout = ObjectLayout()
        act.actuation.layout.dim = [
            ObjectParameters(object='left_arm0', labels=['dist', 'angle'], size=16, stride=8, size_stride_units='bytes'),
            ObjectParameters(object='right_arm0', labels=['dist', 'angle'], size=16, stride=8, size_stride_units='bytes'),
        ]
        act.actuation.layout.data_offset = 0
        act.actuation.data = [
            0.9605706244711711,
            0.7268366284761449,
            0.15277657411350176,
            0.15874396914567307,
        ]
        act.actuation.is_valid = [True] * len(act.actuation.data)
        act.policy_id = 0
        input_episode.action = act

        new_p = Perception()
        new_p.layout = ObjectLayout()
        new_p.layout.dim = [
            ObjectParameters(object='ball_angle0', labels=['data'], size=8, stride=8, size_stride_units='bytes'),
            ObjectParameters(object='box_angle0', labels=['data'], size=8, stride=8, size_stride_units='bytes'),
            ObjectParameters(object='dist_ball_box0', labels=['distance', 'angle_cos', 'angle_sin'], size=24, stride=8, size_stride_units='bytes'),
            ObjectParameters(object='dist_left_arm_ball0', labels=['distance', 'angle_cos', 'angle_sin'], size=24, stride=8, size_stride_units='bytes'),
            ObjectParameters(object='dist_right_arm_ball0', labels=['distance', 'angle_cos', 'angle_sin'], size=24, stride=8, size_stride_units='bytes'),
        ]
        new_p.layout.data_offset = 0
        new_p.data = [
            0.13624948354867789,
            0.5608655958909254,
            0.5014107877258939,
            0.8669652112813435,
            0.16038767144104005,
            0.28593014958874147,
            0.7707294754836551,
            0.9203635939330914,
            0.7160675264112012,
            0.09700901980790094,
            0.20402995103590088,
        ]
        new_p.is_valid = [True] * len(new_p.data)
        input_episode.perception = new_p

        rl = RewardList()
        rl.goals = ['novelty_goal', 'grasped_ball_drive', 'object_in_box_drive']
        rl.rewards = [0.0, 0.0, 0.0]
        input_episode.reward_list = rl

        input_episode.timestamp = self.get_clock().now().to_msg()

        try:
            self.get_logger().info(
                "[proxy] calling world model predict with 1 episode..."
            )
            wm_result = self.world_model_predict_client.send_request(
                input_episodes=[input_episode]
            )
        except Exception as exc:
            self.get_logger().error(
                f"[proxy] world model predict failed: {exc}"
            )
            response.output_episodes = []
            response.valid = False
            return response

        output_episodes = list(getattr(wm_result, "output_episodes", []))
        response.output_episodes = output_episodes
        response.valid = getattr(wm_result, "valid", True)

        self.get_logger().info(
            f"[proxy] world model result.valid={response.valid}, "
            f"n_output_episodes={len(output_episodes)}"
        )

        if not output_episodes:
            self.get_logger().warning("[proxy] no output episodes returned")
            return response

        episode = output_episodes[0]
        predicted_perception_msg = episode.perception
        predicted_action_msg = episode.action.actuation

        predicted_perception = list(getattr(predicted_perception_msg, "data", []))
        predicted_action = list(getattr(predicted_action_msg, "data", []))

        self.get_logger().info(
            f"[proxy] predicted perception ({len(predicted_perception)}): "
            f"{predicted_perception}"
        )
        self.get_logger().info(
            f"[proxy] predicted action ({len(predicted_action)}): "
            f"{predicted_action}"
        )

        try:
            self.get_logger().info("[proxy] calling utility model predict...")
            utility_episode = EpisodeMsg()
            utility_episode.old_perception = input_episode.old_perception
            utility_episode.perception = predicted_perception_msg
            utility_episode.action = episode.action
            utility_episode.reward_list = RewardList()
            utility_episode.reward_list.goals = []
            utility_episode.reward_list.rewards = []
            utility_episode.parent_policy = self.name
            utility_episode.timestamp = self.get_clock().now().to_msg()

            utility_result = self.utility_model_client.send_request(
                input_episodes=[utility_episode]
            )
        except Exception as exc:
            self.get_logger().error(f"[proxy] utility model predict failed: {exc}")
            return response

        expected_utilities = list(getattr(utility_result, "expected_utilities", []))
        self.get_logger().info(
            f"[proxy] utility model result.valid={getattr(utility_result, 'valid', False)}, "
            f"expected_utilities={expected_utilities}"
            )
        if not getattr(utility_result, "valid", False):
            self.get_logger().warning("[proxy] utility model returned invalid result")
            predicted_utility = 0.0
        else:
            expected_utilities = list(getattr(utility_result, "expected_utilities", []))
            if not expected_utilities:
                self.get_logger().warning("[proxy] utility model returned no expected_utilities")
                predicted_utility = 0.0
            else:
                predicted_utility = float(expected_utilities[0])

        predicted_utility = float(predicted_utility)

        utility_rl = RewardList()
        utility_rl.goals = ["utility_model"]
        utility_rl.rewards = [predicted_utility]
        episode.reward_list = utility_rl

        response.output_episodes = [episode]
        return response

    async def execute_callback(self, request, response):
        self.get_logger().info("Executing dummy policy (execute_callback)...")
        response.policy = self.name
        return response

    def _on_world_model_predict_done(self, future):
        try:
            result = future.result()
            output_episodes = list(getattr(result, "output_episodes", []))
            self.get_logger().info(
                f"World model predict done: n_output_episodes={len(output_episodes)}"
            )
            if output_episodes:
                pe = output_episodes[0]
                self.get_logger().info(
                    f"Predicted perception length={len(pe.perception.data)} "
                    f"values={list(pe.perception.data)}"
                )
        except Exception as exc:
            self.get_logger().error(f"World model predict callback failed: {exc}")


class PolicyRandomAction(PolicyBlocking):
    def __init__(self, name='policy_random_action', actuation_config=None, **params):
        super().__init__(name, **params)
        self.actuation_config = actuation_config
        self.actuation = {}
        self.setup()

    def setup(self):
        self.world_model_client = ServiceClientAsync(
            self,
            Predict,
            "/world_model/GRIPPER_AND_LOW_FRICTION/predict",
            self.cbgroup_client
        )
        random_seed = getattr(self, 'random_seed', None)
        self.rng = np.random.default_rng(random_seed)
        for actuator in self.actuation_config:
            self.actuation[actuator] = [{}]
            for param in self.actuation_config[actuator]:
                if self.actuation_config[actuator][param]["type"] == "float":
                    self.actuation[actuator][0][param] = 0.0
                elif self.actuation_config[actuator][param]["type"] == "bool":
                    self.actuation[actuator][0][param] = False
                else:
                    raise TypeError("Type assigned to actuator not recognized")

    def randomize_actuation(self):
        for actuator in self.actuation:
            for param in self.actuation[actuator][0]:
                if self.actuation_config[actuator][param]["type"] == "float":
                    self.actuation[actuator][0][param] = self.rng.uniform()
                elif self.actuation_config[actuator][param]["type"] == "bool":
                    self.actuation[actuator][0][param] = self.rng.choice(
                        [True, True, True, True, True, False]
                    )
                else:
                    self.get_logger().info(
                        f"DEBUG: {actuator}, {param} {self.actuation[actuator][0][param]} "
                        f"type: {type(self.actuation[actuator][0][param])}"
                    )
                    raise TypeError("Actuation parameter is of unknown type")
                self.get_logger().info(
                    f"DEBUG: {actuator}, {param} : {self.actuation[actuator][0][param]}"
                )

    async def execute_callback(self, request, response):
        self.get_logger().info('Executing policy: ' + self.name + '...')
        self.randomize_actuation()
        actuation_msg = actuation_dict_to_msg(self.actuation)
        input_episode = EpisodeMsg()
        input_episode.old_perception = request.perception
        input_episode.action.actuation = actuation_msg
        await self.world_model_client.send_request_async(input_episodes=[input_episode])
        await self.policy_service.send_request_async(action=actuation_msg)
        response.policy = self.name
        response.action = actuation_dict_to_msg(self.actuation)
        return response