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

from core.utils import actuation_dict_to_msg, resolve_seed
from core_interfaces.srv import GetNodeFromLTM

from cognitive_node_interfaces.msg import Episode as EpisodeMsg
from cognitive_node_interfaces.msg import SuccessRate
from cognitive_node_interfaces.msg import Perception
from cognitive_node_interfaces.msg import Action as ActionMsg
from cognitive_node_interfaces.srv import Execute, Predict, PredictUtility
from cognitive_node_interfaces.msg import ObjectLayout, Actuation, ObjectParameters
from cognitive_node_interfaces.srv import GetDreamingSelection

from cognitive_processes_interfaces.msg import RewardList

from cognitive_nodes.drive import Drive
from cognitive_nodes.goal import Goal
from cognitive_nodes.policy import Policy, PolicyBlocking
from cognitive_processes.cognitive_process import CognitiveProcess
from cognitive_nodes.episode import Episode, Action as EpisodeAction


from collections import deque
from functools import partial
import threading
import traceback
from copy import deepcopy

from core.service_client import ServiceClient, ServiceClientAsync

import rclpy
import numpy as np
import yaml

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC

from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import String

from core.utils import actuation_dict_to_msg, resolve_seed
from core_interfaces.srv import GetNodeFromLTM

from cognitive_node_interfaces.msg import Episode as EpisodeMsg
from cognitive_node_interfaces.msg import SuccessRate
from cognitive_node_interfaces.msg import Perception
from cognitive_node_interfaces.msg import Action as ActionMsg
from cognitive_node_interfaces.srv import Execute, Predict, PredictUtility
from cognitive_node_interfaces.msg import ObjectLayout, Actuation, ObjectParameters
from cognitive_node_interfaces.srv import GetDreamingSelection

from cognitive_processes_interfaces.msg import RewardList

from cognitive_nodes.drive import Drive
from cognitive_nodes.goal import Goal
from cognitive_nodes.policy import Policy, PolicyBlocking
from cognitive_processes.cognitive_process import CognitiveProcess
from cognitive_nodes.episode import Episode, Action as EpisodeAction


class DreamingSacEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, policy_node, max_steps=25):
        super().__init__()
        self.policy_node = policy_node
        self.max_steps = int(max_steps)
        self.current_step = 0

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.policy_node.perception_size,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.policy_node.action_size,),
            dtype=np.float32,
        )

        self.current_perception_msg = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        if options and "initial_perception_msg" in options and options["initial_perception_msg"] is not None:
            self.current_perception_msg = deepcopy(options["initial_perception_msg"])
        else:
            self.current_perception_msg = self.policy_node.make_zero_perception_msg()

        obs = self.policy_node.perception_msg_to_obs(self.current_perception_msg)
        return obs, {}

    def step(self, action):

        self.current_step += 1

        action_msg = self.policy_node.action_vec_to_action_msg(action)
        predicted_episode_msg, reward = self.policy_node.rollout_with_models(
            old_perception_msg=self.current_perception_msg,
            action_msg=action_msg,
        )

        self.current_perception_msg = deepcopy(predicted_episode_msg.perception)
        next_obs = self.policy_node.perception_msg_to_obs(self.current_perception_msg)

        terminated = False
        truncated = self.current_step >= self.max_steps
        info = {
            "predicted_reward": float(reward),
        }


        return next_obs, float(reward), terminated, truncated, info


class PolicyDreaming(Policy):
    def __init__(self, name="policy_dreaming", dream_process_class=None, **params):
        super().__init__(name, **params)

        self.dreaming = None
        self._sac_lock = threading.Lock()

        self.selected_world_model = None
        self.selected_utility_model = None

        self.world_model_client = None
        self.utility_model_client = None

        self.world_model_predict_service_name = None
        self.utility_model_predict_service_name = None

        self.selection_service_name = "/dreaming_drive/get_selection"
        self.selection_client = None

        self._sac = None
        self._sac_signature = None

        self.declare_parameter("perception_size", 11)
        self.declare_parameter("action_size", 4)
        self.declare_parameter("sac_total_timesteps", 50)
        self.declare_parameter("sac_learning_starts", 10)
        self.declare_parameter("sac_batch_size", 16)
        self.declare_parameter("sac_gamma", 0.99)
        self.declare_parameter("sac_tau", 0.005)
        self.declare_parameter("sac_train_freq", 1)
        self.declare_parameter("sac_gradient_steps", 1)
        self.declare_parameter("dream_episode_len", 5)
        self.declare_parameter("sac_verbose", 1)
        self.perception_size = self.get_parameter(
            "perception_size"
        ).get_parameter_value().integer_value

        self.action_size = self.get_parameter(
            "action_size"
        ).get_parameter_value().integer_value

        self.sac_total_timesteps = self.get_parameter(
            "sac_total_timesteps"
        ).get_parameter_value().integer_value

        self.sac_learning_starts = self.get_parameter(
            "sac_learning_starts"
        ).get_parameter_value().integer_value

        self.sac_batch_size = self.get_parameter(
            "sac_batch_size"
        ).get_parameter_value().integer_value

        self.sac_gamma = self.get_parameter(
            "sac_gamma"
        ).get_parameter_value().double_value

        self.sac_tau = self.get_parameter(
            "sac_tau"
        ).get_parameter_value().double_value

        self.sac_train_freq = self.get_parameter(
            "sac_train_freq"
        ).get_parameter_value().integer_value

        self.sac_gradient_steps = self.get_parameter(
            "sac_gradient_steps"
        ).get_parameter_value().integer_value

        self.dream_episode_len = self.get_parameter(
            "dream_episode_len"
        ).get_parameter_value().integer_value

        self.sac_verbose = self.get_parameter(
            "sac_verbose"
        ).get_parameter_value().integer_value

        if dream_process_class is not None:
            self.dreaming = dream_process_class(
                name=f"{self.name}_process",
                node=self,
                **params
            )

        self.get_logger().info("PolicyDreaming initialized")

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------

    def _make_layout(self):
        layout = ObjectLayout()
        layout.dim = []
        layout.data_offset = 0
        return layout

    def _perception_layout(self):
        layout = ObjectLayout()
        layout.dim = [
            ObjectParameters(
                object='ball_angle0',
                labels=['data'],
                size=8,
                stride=8,
                size_stride_units='bytes'
            ),
            ObjectParameters(
                object='box_angle0',
                labels=['data'],
                size=8,
                stride=8,
                size_stride_units='bytes'
            ),
            ObjectParameters(
                object='dist_ball_box0',
                labels=['distance', 'angle_cos', 'angle_sin'],
                size=24,
                stride=8,
                size_stride_units='bytes'
            ),
            ObjectParameters(
                object='dist_left_arm_ball0',
                labels=['distance', 'angle_cos', 'angle_sin'],
                size=24,
                stride=8,
                size_stride_units='bytes'
            ),
            ObjectParameters(
                object='dist_right_arm_ball0',
                labels=['distance', 'angle_cos', 'angle_sin'],
                size=24,
                stride=8,
                size_stride_units='bytes'
            ),
        ]
        layout.data_offset = 0
        return layout

    def _action_layout(self):
        layout = ObjectLayout()
        layout.dim = [
            ObjectParameters(
                object='left_arm0',
                labels=['dist', 'angle'],
                size=16,
                stride=8,
                size_stride_units='bytes'
            ),
            ObjectParameters(
                object='right_arm0',
                labels=['dist', 'angle'],
                size=16,
                stride=8,
                size_stride_units='bytes'
            ),
        ]
        layout.data_offset = 0
        return layout

    # ------------------------------------------------------------------
    # Message builders / converters
    # ------------------------------------------------------------------

    def make_zero_perception_msg(self) -> Perception:
        p = Perception()
        p.layout = self._perception_layout()
        p.data = [0.0] * self.perception_size
        p.is_valid = [True] * self.perception_size
        return p

    def perception_msg_to_obs(self, perception_msg: Perception) -> np.ndarray:
        data = list(getattr(perception_msg, "data", []))
        if len(data) != self.perception_size:
            if len(data) < self.perception_size:
                data = data + [0.0] * (self.perception_size - len(data))
            else:
                data = data[:self.perception_size]
        return np.asarray(data, dtype=np.float32)

    def dict_perception_to_msg(self, perception_dict) -> Perception:
        if isinstance(perception_dict, Perception):
            return deepcopy(perception_dict)

        if not isinstance(perception_dict, dict):
            raise TypeError("perception_dict must be dict or Perception")

        def safe_get(d, key, subkey, default=0.0):
            try:
                value = d[key][0][subkey]
                return float(value)
            except Exception:
                return float(default)

        values = [
            safe_get(perception_dict, "ball_angle", "data", 0.0),
            safe_get(perception_dict, "box_angle", "data", 0.0),
            safe_get(perception_dict, "dist_ball_box", "distance", 0.0),
            safe_get(perception_dict, "dist_ball_box", "angle_cos", 0.0),
            safe_get(perception_dict, "dist_ball_box", "angle_sin", 0.0),
            safe_get(perception_dict, "dist_left_arm_ball", "distance", 0.0),
            safe_get(perception_dict, "dist_left_arm_ball", "angle_cos", 0.0),
            safe_get(perception_dict, "dist_left_arm_ball", "angle_sin", 0.0),
            safe_get(perception_dict, "dist_right_arm_ball", "distance", 0.0),
            safe_get(perception_dict, "dist_right_arm_ball", "angle_cos", 0.0),
            safe_get(perception_dict, "dist_right_arm_ball", "angle_sin", 0.0),
        ]

        p = Perception()
        p.layout = self._perception_layout()
        p.data = [float(v) for v in values]
        p.is_valid = [True] * len(p.data)
        return p

    def perception_msg_to_dict(self, perception_msg: Perception):
        data = list(getattr(perception_msg, "data", []))
        if len(data) < 11:
            data = data + [0.0] * (11 - len(data))

        return {
            "ball_angle": [{"data": float(data[0])}],
            "box_angle": [{"data": float(data[1])}],
            "dist_ball_box": [{
                "distance": float(data[2]),
                "angle_cos": float(data[3]),
                "angle_sin": float(data[4]),
            }],
            "dist_left_arm_ball": [{
                "distance": float(data[5]),
                "angle_cos": float(data[6]),
                "angle_sin": float(data[7]),
            }],
            "dist_right_arm_ball": [{
                "distance": float(data[8]),
                "angle_cos": float(data[9]),
                "angle_sin": float(data[10]),
            }],
        }

    def action_vec_to_action_msg(self, action_vec) -> ActionMsg:
        vec = np.asarray(action_vec, dtype=np.float32).reshape(-1)
        if len(vec) != self.action_size:
            raise RuntimeError(
                f"Expected action vector of size {self.action_size}, got {len(vec)}"
            )

        vec = np.clip(vec, 0.0, 1.0)

        act = ActionMsg()
        act.actuation = Actuation()
        act.actuation.layout = self._action_layout()
        act.actuation.data = [float(v) for v in vec.tolist()]
        act.actuation.is_valid = [True] * len(act.actuation.data)
        act.policy_id = 0
        return act

    def action_vec_to_dict(self, action_vec):
        vec = np.asarray(action_vec, dtype=np.float32).reshape(-1)
        if len(vec) != self.action_size:
            raise RuntimeError(
                f"Expected action vector of size {self.action_size}, got {len(vec)}"
            )

        vec = np.clip(vec, 0.0, 1.0)

        return {
            "left_arm": [{
                "dist": float(vec[0]),
                "angle": float(vec[1]),
            }],
            "right_arm": [{
                "dist": float(vec[2]),
                "angle": float(vec[3]),
            }],
        }

    def dict_action_to_action_msg(self, actuation_dict) -> ActionMsg:
        left = actuation_dict.get("left_arm", [{}])[0]
        right = actuation_dict.get("right_arm", [{}])[0]

        vec = np.array([
            float(left.get("dist", 0.0)),
            float(left.get("angle", 0.0)),
            float(right.get("dist", 0.0)),
            float(right.get("angle", 0.0)),
        ], dtype=np.float32)

        return self.action_vec_to_action_msg(vec)

    # ------------------------------------------------------------------
    # SAC policy action
    # ------------------------------------------------------------------

    def select_action_with_sac(self, perception):
        if self._sac is None:
            raise RuntimeError("SAC model is not available in PolicyDreaming")

        if isinstance(perception, dict):
            perception_msg = self.dict_perception_to_msg(perception)
        elif isinstance(perception, Perception):
            perception_msg = perception
        else:
            raise TypeError("perception must be dict or Perception")

        obs = self.perception_msg_to_obs(perception_msg)

        with self._sac_lock:
            action_vec, _ = self._sac.predict(obs, deterministic=False)

        self.get_logger().info(
            f"SAC dream action: obs_dim={len(obs)}, action={np.array(action_vec).tolist()}"
        )

        return self.action_vec_to_dict(action_vec)

    # ------------------------------------------------------------------
    # Service/client setup
    # ------------------------------------------------------------------

    def _ensure_selection_client(self):
        if self.selection_client is not None:
            return

        tmp_client = self.create_client(
            GetDreamingSelection,
            self.selection_service_name
        )
        if not tmp_client.wait_for_service(timeout_sec=2.0):
            raise RuntimeError(
                f"{self.selection_service_name} not available yet"
            )

        self.selection_client = ServiceClient(
            GetDreamingSelection,
            self.selection_service_name
        )

    def _sync_selected_models(self):
        try:
            self._ensure_selection_client()
            response = self.selection_client.send_request()
        except Exception as exc:
            self.get_logger().error(
                f"[PolicyDreaming] selection service call failed: {exc}"
            )
            return None, None

        self.selected_world_model = response.world_model or None
        self.selected_utility_model = response.utility_model or None

        self.get_logger().info(
            f"[PolicyDreaming] selection from drive: "
            f"world_model={self.selected_world_model}, "
            f"utility_model={self.selected_utility_model}, "
            f"active={response.active}"
        )

        return self.selected_world_model, self.selected_utility_model

    def _ensure_world_model_client(self, world_model_name):
        if not world_model_name:
            return None

        service_name = f"/world_model/{world_model_name}/predict"

        if (
            self.world_model_client is not None
            and self.world_model_predict_service_name == service_name
        ):
            return self.world_model_client

        tmp_client = self.create_client(Predict, service_name)
        if not tmp_client.wait_for_service(timeout_sec=2.0):
            raise RuntimeError(f"{service_name} not available yet")

        self.world_model_predict_service_name = service_name
        self.world_model_client = ServiceClient(Predict, service_name)

        self.get_logger().info(
            f"[PolicyDreaming] world model client configured for {service_name}"
        )
        return self.world_model_client

    def _ensure_utility_model_client(self, utility_model_name):
        if not utility_model_name:
            return None

        service_name = f"/utility_model/{utility_model_name}/predict"

        if (
            self.utility_model_client is not None
            and self.utility_model_predict_service_name == service_name
        ):
            return self.utility_model_client

        tmp_client = self.create_client(PredictUtility, service_name)
        if not tmp_client.wait_for_service(timeout_sec=2.0):
            raise RuntimeError(f"{service_name} not available yet")

        self.utility_model_predict_service_name = service_name
        self.utility_model_client = ServiceClient(PredictUtility, service_name)

        self.get_logger().info(
            f"[PolicyDreaming] utility model client configured for {service_name}"
        )
        return self.utility_model_client

    # ------------------------------------------------------------------
    # World model + utility model rollout
    # ------------------------------------------------------------------

    def rollout_with_models(self, old_perception_msg: Perception, action_msg: ActionMsg):
        if self.world_model_client is None:
            raise RuntimeError("world_model_client is not configured")
        if self.utility_model_client is None:
            raise RuntimeError("utility_model_client is not configured")

        input_episode = EpisodeMsg()
        input_episode.old_perception = deepcopy(old_perception_msg)
        input_episode.parent_policy = self.name
        input_episode.action = deepcopy(action_msg)
        input_episode.reward_list = RewardList()
        input_episode.reward_list.goals = []
        input_episode.reward_list.rewards = []
        input_episode.timestamp = self.get_clock().now().to_msg()

        wm_result = self.world_model_client.send_request(
            input_episodes=[input_episode]
        )

        # if not getattr(wm_result, "valid", True):
        #     raise RuntimeError("World model returned invalid=False")

        output_episodes = list(getattr(wm_result, "output_episodes", []))
        if not output_episodes:
            raise RuntimeError("World model returned no output_episodes")

        predicted_episode = output_episodes[0]
        predicted_perception_msg = predicted_episode.perception

        utility_episode = EpisodeMsg()
        utility_episode.old_perception = deepcopy(old_perception_msg)
        utility_episode.perception = deepcopy(predicted_perception_msg)
        utility_episode.action = deepcopy(predicted_episode.action)
        utility_episode.reward_list = RewardList()
        utility_episode.reward_list.goals = []
        utility_episode.reward_list.rewards = []
        utility_episode.parent_policy = self.name
        utility_episode.timestamp = self.get_clock().now().to_msg()

        utility_result = self.utility_model_client.send_request(
            input_episodes=[utility_episode]
        )

        if not getattr(utility_result, "valid", False):
            predicted_utility = 0.0
        else:
            expected_utilities = list(
                getattr(utility_result, "expected_utilities", [])
            )
            predicted_utility = (
                float(expected_utilities[0]) if expected_utilities else 0.0
            )

        utility_rl = RewardList()
        utility_rl.goals = ["utility_model"]
        utility_rl.rewards = [float(predicted_utility)]
        predicted_episode.reward_list = utility_rl

        return predicted_episode, float(predicted_utility)

    # ------------------------------------------------------------------
    # API for Dreaming process
    # ------------------------------------------------------------------

    def predict_with_world_model(self, old_perception, action):
        if isinstance(old_perception, dict):
            old_perception_msg = self.dict_perception_to_msg(old_perception)
        elif isinstance(old_perception, Perception):
            old_perception_msg = old_perception
        else:
            raise TypeError("old_perception must be dict or Perception")

        if isinstance(action, EpisodeAction):
            action_msg = self.dict_action_to_action_msg(action.actuation)
        elif isinstance(action, ActionMsg):
            action_msg = action
        else:
            raise TypeError("action must be EpisodeAction or ActionMsg")

        predicted_episode, _ = self.rollout_with_models(
            old_perception_msg=old_perception_msg,
            action_msg=action_msg,
        )
        return self.perception_msg_to_dict(predicted_episode.perception)

    def evaluate_with_utility_model(self, old_perception, new_perception):
        if isinstance(old_perception, dict):
            old_perception_msg = self.dict_perception_to_msg(old_perception)
        elif isinstance(old_perception, Perception):
            old_perception_msg = old_perception
        else:
            raise TypeError("old_perception must be dict or Perception")

        if isinstance(new_perception, dict):
            new_perception_msg = self.dict_perception_to_msg(new_perception)
        elif isinstance(new_perception, Perception):
            new_perception_msg = new_perception
        else:
            raise TypeError("new_perception must be dict or Perception")

        utility_episode = EpisodeMsg()
        utility_episode.old_perception = deepcopy(old_perception_msg)
        utility_episode.perception = deepcopy(new_perception_msg)
        utility_episode.action = ActionMsg()
        utility_episode.action.actuation = Actuation()
        utility_episode.action.actuation.layout = self._action_layout()
        utility_episode.action.actuation.data = [0.0] * self.action_size
        utility_episode.action.actuation.is_valid = [True] * self.action_size
        utility_episode.action.policy_id = 0
        utility_episode.reward_list = RewardList()
        utility_episode.reward_list.goals = []
        utility_episode.reward_list.rewards = []
        utility_episode.parent_policy = self.name
        utility_episode.timestamp = self.get_clock().now().to_msg()

        utility_result = self.utility_model_client.send_request(
            input_episodes=[utility_episode]
        )

        if not getattr(utility_result, "valid", False):
            return 0.0

        expected_utilities = list(getattr(utility_result, "expected_utilities", []))
        return float(expected_utilities[0]) if expected_utilities else 0.0

    # ------------------------------------------------------------------
    # SAC training
    # ------------------------------------------------------------------

    def train_sac(self, world_model_name, utility_model_name):
        self.get_logger().info(
            f"[PolicyDreaming] train_sac start | "
            f"world_model={world_model_name}, utility_model={utility_model_name}"
        )

        env = DreamingSacEnv(
            policy_node=self,
            max_steps=self.dream_episode_len,
        )

        with self._sac_lock:
            self._sac = SAC(
                policy="MlpPolicy",
                env=env,
                learning_starts=self.sac_learning_starts,
                batch_size=self.sac_batch_size,
                gamma=self.sac_gamma,
                tau=self.sac_tau,
                train_freq=self.sac_train_freq,
                gradient_steps=self.sac_gradient_steps,
                verbose=self.sac_verbose,
            )

            self._sac.learn(total_timesteps=self.sac_total_timesteps)

        self.get_logger().info(
            f"[PolicyDreaming] train_sac done | total_timesteps={self.sac_total_timesteps}"
        )

    def _ensure_sac_trained(self, selected_world_model, selected_utility_model):
        current_signature = (selected_world_model, selected_utility_model)

        if (
            getattr(self, "_sac", None) is not None
            and getattr(self, "_sac_signature", None) == current_signature
        ):
            self.get_logger().info(
                f"[PolicyDreaming] SAC already trained for {current_signature}"
            )
            return

        self.get_logger().info(
            f"[PolicyDreaming] Training SAC for "
            f"world_model={selected_world_model}, "
            f"utility_model={selected_utility_model}"
        )

        self.train_sac(
            world_model_name=selected_world_model,
            utility_model_name=selected_utility_model,
        )

        self._sac_signature = current_signature

        self.get_logger().info(
            f"[PolicyDreaming] SAC training finished for {current_signature}"
        )

    # ------------------------------------------------------------------
    # Execute callback
    # ------------------------------------------------------------------

    async def execute_callback(self, request, response):
        selected_world_model, selected_utility_model = self._sync_selected_models()

        if selected_world_model is None:
            self.get_logger().warning("No selected world model available")
            response.policy = self.name
            return response

        if selected_utility_model is None:
            self.get_logger().warning("No selected utility model available")
            response.policy = self.name
            return response

        try:
            self._ensure_world_model_client(selected_world_model)
            self._ensure_utility_model_client(selected_utility_model)

            self._ensure_sac_trained(
                selected_world_model=selected_world_model,
                selected_utility_model=selected_utility_model,
            )

        except Exception as exc:
            self.get_logger().error(
                f"[PolicyDreaming] failed during setup/training: {exc}"
            )
            response.policy = self.name
            return response

        if self.dreaming is not None:
            if hasattr(request, "perception") and request.perception is not None:
                try:
                    self.dreaming.initial_perception = self.perception_msg_to_dict(
                        request.perception
                    )
                except Exception:
                    self.dreaming.initial_perception = None

            self.dreaming.start_flag.set()

        response.policy = self.name
        self.get_logger().info(
            f"PolicyDreaming ready with world_model={selected_world_model}, "
            f"utility_model={selected_utility_model}"
        )
        return response


class DriveDreaming(Drive):
    def __init__(
        self,
        name="dreaming_drive",
        class_name="cognitive_nodes.drive.Drive",
        ltm_service_name="ltm_0/get_node",
        prediction_error_threshold=0.02,
        discovery_period=5.0,
        candidate_world_models=None,
        candidate_utility_models=None,
        preferred_utility_model=None,
        **params
    ):
        super().__init__(name, class_name, **params)

        self.prediction_error_threshold = prediction_error_threshold
        self.discovery_period = discovery_period

        self.candidate_world_models = candidate_world_models or []
        self.candidate_utility_models = candidate_utility_models or []
        self.preferred_utility_model = preferred_utility_model

        self.known_world_models = set()
        self.known_utility_models = set()

        self.pending_ltm_queries = set()
        self.full_ltm_query_pending = False

        self.world_model_subscribers = {}
        self.world_model_errors = {}

        self.world_model_predict_clients = {}
        self.utility_model_predict_clients = {}

        self.selected_world_model = None
        self.selected_utility_model = None

        self.ltm_client = self.create_client(GetNodeFromLTM, ltm_service_name)

        self.selection_srv = self.create_service(
        GetDreamingSelection,
        "/dreaming_drive/get_selection",
        self._handle_get_selection,
        )

        self.get_logger().debug(
            f"[DriveDreaming:init] threshold={self.prediction_error_threshold}, "
            f"discovery_period={self.discovery_period}, "
            f"candidate_world_models={self.candidate_world_models}, "
            f"candidate_utility_models={self.candidate_utility_models}, "
            f"preferred_utility_model={self.preferred_utility_model}, "
            f"ltm_service_name={ltm_service_name}"
        )

        self.discovery_timer = self.create_timer(
            self.discovery_period,
            self.refresh_models_from_ltm
        )


    def _handle_get_selection(self, request, response):
        # if no world model or utility model is selected, return empty strings
        world = self.selected_world_model or ""
        utility = self.selected_utility_model or ""

        response.world_model = world
        response.utility_model = utility
        response.active = bool(world and utility)

        self.get_logger().debug(
            f"[DriveDreaming:service] get_selection -> "
            f"world_model={response.world_model}, "
            f"utility_model={response.utility_model}, "
            f"active={response.active}"
        )
        return response

    def refresh_models_from_ltm(self):
        self.get_logger().debug(
            f"[DriveDreaming:refresh] start | known_world_models={sorted(self.known_world_models)} "
            f"| known_utility_models={sorted(self.known_utility_models)} "
            f"| selected_world_model={self.selected_world_model} "
            f"| selected_utility_model={self.selected_utility_model} "
            f"| pending={sorted(self.pending_ltm_queries)} "
            f"| full_ltm_query_pending={self.full_ltm_query_pending}"
        )

        if not self.ltm_client.service_is_ready():
            self.get_logger().warning("[DriveDreaming:refresh] LTM service not ready yet")
            return

        explicit_candidates = set(self.candidate_world_models) | set(self.candidate_utility_models)
        if explicit_candidates:
            for node_name in explicit_candidates:
                if node_name in self.pending_ltm_queries:
                    self.get_logger().debug(f"[DriveDreaming:refresh] explicit query already pending for {node_name}")
                    continue

                already_known = (
                    node_name in self.known_world_models or
                    node_name in self.known_utility_models
                )
                if already_known:
                    self.get_logger().debug(f"[DriveDreaming:refresh] explicit candidate already known: {node_name}")
                    continue

                self.get_logger().debug(f"[DriveDreaming:refresh] querying LTM for explicit candidate: {node_name}")
                req = GetNodeFromLTM.Request()
                req.name = node_name
                future = self.ltm_client.call_async(req)
                self.pending_ltm_queries.add(node_name)
                future.add_done_callback(partial(self.handle_named_ltm_response, queried_name=node_name))
            return

        if self.full_ltm_query_pending:
            self.get_logger().debug("[DriveDreaming:refresh] full LTM dump already pending")
            return

        self.get_logger().debug("[DriveDreaming:refresh] requesting full LTM dump")
        req = GetNodeFromLTM.Request()
        req.name = ""
        future = self.ltm_client.call_async(req)
        self.full_ltm_query_pending = True
        future.add_done_callback(self.handle_full_ltm_response)

    def handle_named_ltm_response(self, future, queried_name):
        self.pending_ltm_queries.discard(queried_name)

        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f"[DriveDreaming:named] LTM query failed for {queried_name}: {e}")
            return

        raw_data = response.data
        self.get_logger().debug(f"[DriveDreaming:named] raw response for {queried_name}: {str(raw_data)[:500]}")

        try:
            node_data = yaml.safe_load(raw_data)
        except Exception as e:
            self.get_logger().error(f"[DriveDreaming:named] YAML parse failed for {queried_name}: {e}")
            return

        is_wm = self.is_world_model(node_data)
        is_um = self.is_utility_model(node_data)

        self.get_logger().debug(
            f"[DriveDreaming:named] queried_name={queried_name} "
            f"| is_world_model={is_wm} | is_utility_model={is_um}"
        )

        if is_wm:
            self.register_world_model(queried_name)

        if is_um:
            self.register_utility_model(queried_name)

        if not is_wm and not is_um:
            self.get_logger().warning(f"[DriveDreaming:named] {queried_name} is neither WorldModel nor UtilityModel")

        self.log_current_selection_state()

    def handle_full_ltm_response(self, future):
        self.full_ltm_query_pending = False

        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f"[DriveDreaming:full] full LTM dump query failed: {e}")
            return

        raw_data = response.data
        self.get_logger().debug(f"[DriveDreaming:full] full LTM dump received: {str(raw_data)[:500]}")

        try:
            ltm_data = yaml.safe_load(raw_data)
        except Exception as e:
            self.get_logger().error(f"[DriveDreaming:full] YAML parse failed for full LTM dump: {e}")
            return

        discovered_world_models, discovered_utility_models = self.extract_model_names(ltm_data)

        self.get_logger().debug(
            f"[DriveDreaming:full] discovered world models={discovered_world_models} "
            f"| discovered utility models={discovered_utility_models}"
        )

        for world_model_name in discovered_world_models:
            self.register_world_model(world_model_name)

        for utility_model_name in discovered_utility_models:
            self.register_utility_model(utility_model_name)

        self.log_current_selection_state()

    def extract_model_names(self, data):
        discovered_world_models = set()
        discovered_utility_models = set()

        if isinstance(data, dict):
            world_section = data.get("WorldModel")
            if isinstance(world_section, dict):
                self.get_logger().debug(
                    f"[DriveDreaming:extract] WorldModel section keys: {list(world_section.keys())}"
                )
                for name in world_section.keys():
                    discovered_world_models.add(str(name))

            utility_section = data.get("UtilityModel")
            if isinstance(utility_section, dict):
                self.get_logger().debug(
                    f"[DriveDreaming:extract] UtilityModel section keys: {list(utility_section.keys())}"
                )
                for name in utility_section.keys():
                    discovered_utility_models.add(str(name))

        def visit(obj, path="root"):
            if isinstance(obj, dict):
                node_name = obj.get("name") or obj.get("node_name")
                node_type = obj.get("nodetype") or obj.get("node_type")
                class_name = (
                    obj.get("class_name")
                    or obj.get("defaultclass")
                    or obj.get("default_class")
                )

                if node_name and node_type == "WorldModel":
                    self.get_logger().debug(
                        f"[DriveDreaming:extract] WorldModel by node_type at {path}: {node_name}"
                    )
                    discovered_world_models.add(str(node_name))

                if node_name and class_name and "WorldModel" in str(class_name):
                    self.get_logger().debug(
                        f"[DriveDreaming:extract] WorldModel by class_name at {path}: {node_name}"
                    )
                    discovered_world_models.add(str(node_name))

                if node_name and node_type == "UtilityModel":
                    self.get_logger().debug(
                        f"[DriveDreaming:extract] UtilityModel by node_type at {path}: {node_name}"
                    )
                    discovered_utility_models.add(str(node_name))

                if node_name and class_name and "UtilityModel" in str(class_name):
                    self.get_logger().debug(
                        f"[DriveDreaming:extract] UtilityModel by class_name at {path}: {node_name}"
                    )
                    discovered_utility_models.add(str(node_name))

                for key, value in obj.items():
                    visit(value, f"{path}.{key}")

            elif isinstance(obj, list):
                for index, item in enumerate(obj):
                    visit(item, f"{path}[{index}]")

        visit(data)
        return sorted(discovered_world_models), sorted(discovered_utility_models)
    def is_world_model(self, node_data):
        if isinstance(node_data, dict):
            node_type = node_data.get("nodetype") or node_data.get("node_type")
            class_name = node_data.get("class_name") or node_data.get("defaultclass")
            if node_type == "WorldModel":
                return True
            if class_name and "WorldModel" in str(class_name):
                return True
            return any(self.is_world_model(v) for v in node_data.values())
        if isinstance(node_data, list):
            return any(self.is_world_model(v) for v in node_data)
        return "WorldModel" in str(node_data)

    def is_utility_model(self, node_data):
        if isinstance(node_data, dict):
            node_type = node_data.get("nodetype") or node_data.get("node_type")
            class_name = node_data.get("class_name") or node_data.get("defaultclass")
            if node_type == "UtilityModel":
                return True
            if class_name and "UtilityModel" in str(class_name):
                return True
            return any(self.is_utility_model(v) for v in node_data.values())
        if isinstance(node_data, list):
            return any(self.is_utility_model(v) for v in node_data)
        return "UtilityModel" in str(node_data)

    def register_world_model(self, world_model_name):
        is_new = world_model_name not in self.known_world_models
        self.known_world_models.add(world_model_name)

        if is_new:
            self.get_logger().debug(f"[DriveDreaming:register] registered WorldModel: {world_model_name}")
        else:
            self.get_logger().debug(f"[DriveDreaming:register] WorldModel already registered: {world_model_name}")

        self.ensure_world_model_subscription(world_model_name)
        self.ensure_world_model_predict_client(world_model_name)

        if self.selected_world_model is None and len(self.known_world_models) == 1:
            self.selected_world_model = world_model_name
            self.get_logger().warning(
                f"[DriveDreaming:select] bootstrap selected_world_model={self.selected_world_model} "
                f"(only known world model, waiting for prediction_error refinement)"
            )

    def register_utility_model(self, utility_model_name):
        is_new = utility_model_name not in self.known_utility_models
        self.known_utility_models.add(utility_model_name)

        if is_new:
            self.get_logger().debug(f"[DriveDreaming:register] registered UtilityModel: {utility_model_name}")
        else:
            self.get_logger().debug(f"[DriveDreaming:register] UtilityModel already registered: {utility_model_name}")

        self.ensure_utility_model_predict_client(utility_model_name)
        self.update_selected_utility_model()

    def ensure_world_model_subscription(self, world_model_name):
        topic_name = f"/world_model/{world_model_name}/prediction_error"

        if topic_name in self.world_model_subscribers:
            self.get_logger().debug(f"[DriveDreaming:subscription] already subscribed to {topic_name}")
            return

        self.get_logger().debug(f"[DriveDreaming:subscription] creating subscription to {topic_name}")
        sub = self.create_subscription(
            SuccessRate,
            topic_name,
            lambda msg, wm=world_model_name: self.prediction_error_callback(msg, wm),
            10
        )
        self.world_model_subscribers[topic_name] = sub
        self.get_logger().debug(
            f"[DriveDreaming:subscription] subscribed to {topic_name} "
            f"| all_subscriptions={list(self.world_model_subscribers.keys())}"
        )

    def ensure_world_model_predict_client(self, world_model_name):
        if world_model_name in self.world_model_predict_clients:
            self.get_logger().debug(f"[DriveDreaming:client] world model client already exists for {world_model_name}")
            return

        service_name = f"/world_model/{world_model_name}/predict"
        self.get_logger().debug(f"[DriveDreaming:client] checking world model predict service: {service_name}")

        tmp_client = self.create_client(Predict, service_name)
        if not tmp_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warning(f"[DriveDreaming:client] world model predict service not ready: {service_name}")
            return

        self.world_model_predict_clients[world_model_name] = ServiceClient(Predict, service_name)
        self.get_logger().debug(f"[DriveDreaming:client] created world model client for {service_name}")

    def ensure_utility_model_predict_client(self, utility_model_name):
        if utility_model_name in self.utility_model_predict_clients:
            self.get_logger().debug(f"[DriveDreaming:client] utility model client already exists for {utility_model_name}")
            return

        service_name = f"/utility_model/{utility_model_name}/predict"
        self.get_logger().debug(f"[DriveDreaming:client] checking utility model predict service: {service_name}")

        tmp_client = self.create_client(PredictUtility, service_name)
        if not tmp_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warning(f"[DriveDreaming:client] utility model predict service not ready: {service_name}")
            return

        self.utility_model_predict_clients[utility_model_name] = ServiceClient(PredictUtility, service_name)
        self.get_logger().debug(f"[DriveDreaming:client] created utility model client for {service_name}")

    def prediction_error_callback(self, msg, world_model_name):
        self.get_logger().debug(
            f"[DriveDreaming:prediction_error] world_model={world_model_name} "
            f"| msg.node_name={msg.node_name} "
            f"| msg.node_type={msg.node_type} "
            f"| success_rate={msg.success_rate}"
        )
        self.world_model_errors[world_model_name] = msg.success_rate
        self.get_logger().debug(f"[DriveDreaming:prediction_error] world_model_errors={self.world_model_errors}")
        self.update_selected_world_model()

    def update_selected_world_model(self):
        candidates = {
            name: err
            for name, err in self.world_model_errors.items()
            if err > self.prediction_error_threshold
        }

        previous = self.selected_world_model
        self.get_logger().debug(
            f"[DriveDreaming:select] threshold={self.prediction_error_threshold} "
            f"| candidates_above_threshold={candidates}"
        )

        if candidates:
            self.selected_world_model = max(candidates, key=candidates.get)
        elif len(self.known_world_models) == 1:
            self.selected_world_model = next(iter(self.known_world_models))
        else:
            self.selected_world_model = None

        self.get_logger().warning(
            f"[DriveDreaming:select] selected_world_model changed "
            f"from {previous} to {self.selected_world_model}"
        )

    def update_selected_utility_model(self):
        previous = self.selected_utility_model

        if self.preferred_utility_model and self.preferred_utility_model in self.known_utility_models:
            self.selected_utility_model = self.preferred_utility_model
        elif len(self.known_utility_models) == 1:
            self.selected_utility_model = next(iter(self.known_utility_models))
        elif "novelty_exploration" in self.known_utility_models:
            self.selected_utility_model = "novelty_exploration"
        else:
            self.selected_utility_model = sorted(self.known_utility_models)[0] if self.known_utility_models else None

        self.get_logger().warning(
            f"[DriveDreaming:select] selected_utility_model changed "
            f"from {previous} to {self.selected_utility_model}"
        )

    def get_selected_world_model(self):
        self.get_logger().debug(
            f"[DriveDreaming:get] selected_world_model={self.selected_world_model} "
            f"| known_world_models={sorted(self.known_world_models)}"
        )
        return self.selected_world_model

    def get_selected_utility_model(self):
        self.get_logger().debug(
            f"[DriveDreaming:get] selected_utility_model={self.selected_utility_model} "
            f"| known_utility_models={sorted(self.known_utility_models)}"
        )
        return self.selected_utility_model

    def get_world_model_client(self):
        name = self.get_selected_world_model()
        client = self.world_model_predict_clients.get(name)
        self.get_logger().debug(
            f"[DriveDreaming:get] world_model_client for {name}: {'FOUND' if client else 'MISSING'}"
        )
        return client

    def get_utility_model_client(self):
        name = self.get_selected_utility_model()
        client = self.utility_model_predict_clients.get(name)
        self.get_logger().debug(
            f"[DriveDreaming:get] utility_model_client for {name}: {'FOUND' if client else 'MISSING'}"
        )
        return client

    def log_current_selection_state(self):
        self.get_logger().warning(
            f"[DriveDreaming:state] known_world_models={sorted(self.known_world_models)} "
            f"| world_model_errors={self.world_model_errors} "
            f"| selected_world_model={self.selected_world_model} "
            f"| known_utility_models={sorted(self.known_utility_models)} "
            f"| selected_utility_model={self.selected_utility_model} "
            f"| wm_clients={list(self.world_model_predict_clients.keys())} "
            f"| um_clients={list(self.utility_model_predict_clients.keys())}"
        )

    def evaluate(self, perception=None):
        active = self.selected_world_model is not None and self.selected_utility_model is not None
        self.evaluation.evaluation = 1.0 if active else 0.0
        self.evaluation.timestamp = self.get_clock().now().to_msg()
        if self.selected_world_model is not None and self.selected_utility_model is not None:
            self.get_logger().debug(
                f"[DriveDreaming:evaluate] evaluation={self.evaluation.evaluation} "
                f"| selected_world_model={self.selected_world_model} "
                f"| selected_utility_model={self.selected_utility_model}"
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
        activation_timeout=1.0,
        **params,
    ):
        super().__init__(name, iterations, trials, LTM_id, **params)
        self.node = node
        self.reward_threshold = reward_threshold
        self.activation_timeout = activation_timeout
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

    def update_activations(self, timeout_sec=None):
        timeout_sec = timeout_sec or self.activation_timeout
        self.get_logger().info(
            f"[Dreaming] Updating activations... timeout={timeout_sec}s"
        )

        self.semaphore.acquire()
        try:
            self.activation_time = self.get_clock().now().nanoseconds

            for node in self.activation_inputs:
                self.activation_inputs[node]["flag"].clear()

            for node in self.activation_inputs:
                if node == self.node.name:
                    self.activation_inputs[node]["flag"].set()

                self.get_logger().info(
                    f"[Dreaming] Waiting for activation from node={node}"
                )

                activated = self.activation_inputs[node]["flag"].wait(
                    timeout=timeout_sec
                )

                if not activated:
                    self.get_logger().warning(
                        f"[Dreaming] Activation timeout while waiting for node={node}"
                    )
                    return False

                self.activation_inputs[node]["flag"].clear()

            self.get_logger().info("[Dreaming] All activations received")
            self.get_logger().debug("DEBUG - LTM CACHE: " + str(self.LTM_cache))
            return True

        finally:
            self.semaphore.release()

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
            f"Dreaming {self.get_name()}: completion check - "
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
        self.finished_flag.clear()

        self.get_logger().info(f"Dreaming cycle started for {self.get_name()}")

        self.current_episode.perception = (
            self.initial_perception or self.sample_random_perception_t0()
        )
        self.summary_episode.old_perception = self.current_episode.perception

        activations_ok = self.update_activations(timeout_sec=self.activation_timeout)
        if not activations_ok:
            self.get_logger().warning(
                f"[Dreaming] Aborting cycle due to activation timeout in {self.get_name()}"
            )
            self.finished_flag.set()
            self.start_flag.clear()
            return

        self.summary_episode.old_ltm_state = deepcopy(self.LTM_cache)
        self.active_goals = self.get_linked_goals()

        achieved = False
        step = 0

        while step < self.iterations and not self.stop and not achieved:
            if self.paused:
                step += 1
                continue

            self.get_logger().info(
                f"*** DREAMING STEP: {step + 1}/{self.iterations} ***"
            )

            old_perception = deepcopy(self.current_episode.perception)
            actuation = self.select_dream_action(old_perception)
            self.current_episode.action = EpisodeAction(
                actuation=actuation,
                policy_id=0
            )
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
                f"Dreaming {self.get_name()}: step {step + 1} - "
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


class PolicyDreaming_old(Policy):
    def __init__(self, name="policy_dreaming", dream_process_class=Dreaming, **params):
        super().__init__(name, **params)
        self.dreaming = None
        self._sac_lock = threading.Lock()

        self.selected_world_model = None
        self.selected_utility_model = None

        self.world_model_client = None
        self.utility_model_client = None

        self.world_model_predict_service_name = None
        self.utility_model_predict_service_name = None

        self.selection_service_name = "/dreaming_drive/get_selection"
        self.selection_client = None

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
            "dream_action": [{"value": float(v)} for v in np.atleast_1d(action_vec)]
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

    def _ensure_selection_client(self):
        if self.selection_client is not None:
            return

        tmp_client = self.create_client(
            GetDreamingSelection,
            self.selection_service_name
        )
        if not tmp_client.wait_for_service(timeout_sec=2.0):
            raise RuntimeError(
                f"{self.selection_service_name} not available yet"
            )

        self.selection_client = ServiceClient(
            GetDreamingSelection,
            self.selection_service_name
        )

    def _sync_selected_models(self):
        try:
            self._ensure_selection_client()
            response = self.selection_client.send_request()
        except Exception as exc:
            self.get_logger().error(
                f"[PolicyDreaming] selection service call failed: {exc}"
            )
            return None, None

        self.selected_world_model = response.world_model or None
        self.selected_utility_model = response.utility_model or None

        self.get_logger().info(
            f"[PolicyDreaming] selection from drive: "
            f"world_model={self.selected_world_model}, "
            f"utility_model={self.selected_utility_model}, "
            f"active={response.active}"
        )

        return self.selected_world_model, self.selected_utility_model

    def _ensure_world_model_client(self, world_model_name):
        if not world_model_name:
            return None

        service_name = f"/world_model/{world_model_name}/predict"

        if (
            self.world_model_client is not None
            and self.world_model_predict_service_name == service_name
        ):
            return self.world_model_client

        tmp_client = self.create_client(Predict, service_name)
        if not tmp_client.wait_for_service(timeout_sec=2.0):
            raise RuntimeError(f"{service_name} not available yet")

        self.world_model_predict_service_name = service_name
        self.world_model_client = ServiceClient(Predict, service_name)

        self.get_logger().info(
            f"[PolicyDreaming] world model client configured for {service_name}"
        )
        return self.world_model_client

    def _ensure_utility_model_client(self, utility_model_name):
        if not utility_model_name:
            return None

        service_name = f"/utility_model/{utility_model_name}/predict"

        if (
            self.utility_model_client is not None
            and self.utility_model_predict_service_name == service_name
        ):
            return self.utility_model_client

        tmp_client = self.create_client(PredictUtility, service_name)
        if not tmp_client.wait_for_service(timeout_sec=2.0):
            raise RuntimeError(f"{service_name} not available yet")

        self.utility_model_predict_service_name = service_name
        self.utility_model_client = ServiceClient(PredictUtility, service_name)

        self.get_logger().info(
            f"[PolicyDreaming] utility model client configured for {service_name}"
        )
        return self.utility_model_client
    
    def _ensure_sac_trained(self, selected_world_model, selected_utility_model):
        current_signature = (selected_world_model, selected_utility_model)

        if getattr(self, "_sac", None) is not None and getattr(self, "_sac_signature", None) == current_signature:
            self.get_logger().info(
                f"[PolicyDreaming] SAC already trained for {current_signature}"
            )
            return

        self.get_logger().info(
            f"[PolicyDreaming] Training SAC for "
            f"world_model={selected_world_model}, utility_model={selected_utility_model}"
        )

        self.train_sac(
            world_model_name=selected_world_model,
            utility_model_name=selected_utility_model,
        )

        self._sac_signature = current_signature

        self.get_logger().info(
            f"[PolicyDreaming] SAC training finished for {current_signature}"
        )

    async def execute_callback(self, request, response):
        if self.dreaming is None:
            raise RuntimeError("Dreaming process is not configured")

        selected_world_model, selected_utility_model = self._sync_selected_models()

        if selected_world_model is None:
            self.get_logger().warning("No selected world model available")
            response.policy = self.name
            return response

        if selected_utility_model is None:
            self.get_logger().warning("No selected utility model available")
            response.policy = self.name
            return response

        try:
            self._ensure_world_model_client(selected_world_model)
            self._ensure_utility_model_client(selected_utility_model)

            self._ensure_sac_trained(
                selected_world_model=selected_world_model,
                selected_utility_model=selected_utility_model,
            )

        except Exception as exc:
            self.get_logger().error(
                f"[PolicyDreaming] failed during setup/training: {exc}"
            )
            response.policy = self.name
            return response

        self.dreaming.start_flag.set()

        response.policy = self.name
        self.get_logger().info(
            f"PolicyDreaming ready with world_model={selected_world_model}, "
            f"utility_model={selected_utility_model}"
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

        self.get_logger().debug(
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
        self.get_logger().debug(
            f"World model predict client reset to {self.world_model_predict_service_name}"
        )

    def _recreate_utility_model_client(self):
        self.utility_model_predict_service_name = self._build_utility_model_predict_service_name(
            self.utility_model_name
        )
        self.utility_model_client = None
        self.get_logger().debug(
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

        self.get_logger().debug(
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
            self.get_logger().debug(
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

        self.get_logger().debug(
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

        self.get_logger().debug(
            f"[proxy] predicted perception ({len(predicted_perception)}): "
            f"{predicted_perception}"
        )
        self.get_logger().debug(
            f"[proxy] predicted action ({len(predicted_action)}): "
            f"{predicted_action}"
        )

        try:
            self.get_logger().debug("[proxy] calling utility model predict...")
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
        self.get_logger().debug(
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
        self.get_logger().debug("Executing dummy policy (execute_callback)...")
        response.policy = self.name
        return response

    def _on_world_model_predict_done(self, future):
        try:
            result = future.result()
            output_episodes = list(getattr(result, "output_episodes", []))
            self.get_logger().debug(
                f"World model predict done: n_output_episodes={len(output_episodes)}"
            )
            if output_episodes:
                pe = output_episodes[0]
                self.get_logger().debug(
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
        self.rng = np.random.default_rng(resolve_seed(random_seed))
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
                    self.get_logger().debug(
                        f"DEBUG: {actuator}, {param} {self.actuation[actuator][0][param]} "
                        f"type: {type(self.actuation[actuator][0][param])}"
                    )
                    raise TypeError("Actuation parameter is of unknown type")
                self.get_logger().debug(
                    f"DEBUG: {actuator}, {param} : {self.actuation[actuator][0][param]}"
                )

    async def execute_callback(self, request, response):
        self.get_logger().debug('Executing policy: ' + self.name + '...')
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