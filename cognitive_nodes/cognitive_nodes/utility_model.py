import rclpy
import threading
import numpy as np
from copy import deepcopy
from rclpy.node import Node
from math import isclose
from rclpy.executors import SingleThreadedExecutor

from core.utils import class_from_classname
from cognitive_nodes.episode import Episode, episode_msg_to_obj, episode_msg_list_to_obj_list
from cognitive_nodes.episodic_buffer import EpisodicBuffer, TraceBuffer
from cognitive_processes.deliberation import Deliberation

from cognitive_nodes.deliberative_model import DeliberativeModel, Learner, ANNLearner, Evaluator
from cognitive_node_interfaces.srv import Execute, AddTrace
import pandas as pd


class UtilityModel(DeliberativeModel):
    """
    Utility Model class
    """
    def __init__(self, name='utility_model', class_name = 'cognitive_nodes.utility_model.UtilityModel', prediction_srv_type="cognitive_node_interfaces.srv.PredictUtility", trace_length=20, max_iterations=20, candidate_actions = 5, ltm_id="", **params):
        """
        Constructor of the Utility Model class.

        Initializes a Utility Model instance with the given name and registers it in the LTM.

        :param name: The name of the Utility Model instance.
        :type name: str
        :param class_name: The name of the Utility Model class.
        :type class_name: str
        """
        super().__init__(name, class_name, prediction_srv_type=prediction_srv_type, **params)
        self.configure_activation_inputs(self.neighbors)
        self.setup_model(trace_length=trace_length, max_iterations=max_iterations, candidate_actions=candidate_actions, ltm_id=ltm_id, **params)
        self.execute_service = self.create_service(
            Execute,
            'utility_model/' + str(name) + '/execute',
            self.execute_callback,
            callback_group=self.cbgroup_server
        )

    def setup_model(self, trace_length, max_iterations, candidate_actions, ltm_id, **params):
        """
        Sets up the Utility Model by initializing the episodic buffer, learner, and confidence evaluator.
        """
        self.episodic_buffer = TraceBuffer(self, main_size=trace_length, inputs=['perception'], outputs=[], **params)
        self.learner = DefaultUtilityModelLearner(self, self.episodic_buffer, **params)
        self.confidence_evaluator = DefaultUtilityEvaluator(self, self.learner, self.episodic_buffer, **params)
        self.deliberation = Deliberation(f"{self.name}_deliberation", self, iterations=max_iterations, candidate_actions=candidate_actions, LTM_id=ltm_id, clear_buffer=True, **params)
        self.spin_deliberation()

    def spin_deliberation(self):
        self.deliberation_executor = SingleThreadedExecutor()
        self.deliberation_executor.add_node(self.deliberation)
        self.deliberation_thread = threading.Thread(target=self.deliberation_executor.spin)
        self.deliberation_thread.start()

    def calculate_activation(self, perception = None, activation_list=None):
        """
        Returns the the activation value of the Utility Model.
        Dummy, for the moment, as it returns a random value.

        :param perception: The given perception.
        :type perception: dict
        :return: The activation of the instance.
        :rtype: float
        """
        if activation_list and self.learner.configured:
            self.calculate_activation_max(activation_list)
        else:
            self.activation.activation=0.0
            self.activation.timestamp=self.get_clock().now().to_msg()

    def predict(self, input_episodes: list[Episode]) -> list[float]:
            input_data = self.episodic_buffer.buffer_to_matrix(input_episodes, self.episodic_buffer.input_labels)
            expected_utilities = self.learner.call(input_data)
            self.get_logger().info(f"Predictions: {expected_utilities}")
            return expected_utilities
    
    def execute_callback(self, request, response):
        """
        Callback for the execute service.
        Executes the action and returns the response.

        :param request: The request from the service.
        :type request: cognitive_node_interfaces.srv.Execute.Request
        :param response: The response to be sent back.
        :type response: cognitive_node_interfaces.srv.Execute.Response
        :return: The response with the executed action.
        :rtype: cognitive_node_interfaces.srv.Execute.Response
        """
        self.get_logger().info(f"Executing deliberation: {self.name}")
        self.deliberation.start_flag.set()
        self.deliberation.finished_flag.wait()
        self.deliberation.finished_flag.clear()
        self.get_logger().info(f"Deliberation finished: {self.name}")
        response.policy = self.name
        return response

class NoveltyUtilityModel(UtilityModel):
    """
    Novelty Utility Model class
    This model is used to compute the novelty of the episodes.
    It inherits from the UtilityModel class.
    """
    def __init__(self, name='utility_model', class_name='cognitive_nodes.utility_model.UtilityModel', prediction_srv_type="cognitive_node_interfaces.srv.PredictUtility", trace_length=20, max_iterations=20, candidate_actions=5, min_traces=5, max_traces=50, max_antitraces=10, ltm_id="", **params):
        super().__init__(name=name, class_name=class_name, prediction_srv_type=prediction_srv_type, trace_length=trace_length, max_iterations=max_iterations, candidate_actions=candidate_actions, ltm_id=ltm_id, min_traces=min_traces, max_traces=max_traces, max_antitraces=max_antitraces, **params)

    def setup_model(self, trace_length, max_iterations, candidate_actions, ltm_id, train_traces=1, max_traces=50, **params):
        self.episodic_buffer = TraceBuffer(self, main_size=trace_length, max_traces=max_traces, min_p_traces=train_traces, inputs=['perception'], outputs=[], **params)
        self.learner = NoveltyUtilityModelLearner(self, self.episodic_buffer, **params)
        self.confidence_evaluator = DefaultUtilityEvaluator(self, self.learner, self.episodic_buffer, **params)
        self.deliberation = Deliberation(f"{self.name}_deliberation", self, iterations=max_iterations, candidate_actions=candidate_actions, LTM_id=ltm_id, clear_buffer=False, exploration_process=True, **params)
        self.spin_deliberation()

class HardCodedUtilityModel(UtilityModel):
    """
    Hard Coded Utility Model class
    This model is used to compute the utility of the episodes based on hard coded values.
    It inherits from the UtilityModel class.
    """
    def __init__(self, name='utility_model', class_name='cognitive_nodes.utility_model.UtilityModel', prediction_srv_type="cognitive_node_interfaces.srv.PredictUtility", trace_length=20, max_iterations=20, candidate_actions=5, min_traces=5, max_traces=50, max_antitraces=10, ltm_id="", perception_config=None, **params):
        super().__init__(
            name=name,
            class_name=class_name,
            prediction_srv_type=prediction_srv_type,
            trace_length=trace_length,
            max_iterations=max_iterations,
            candidate_actions=candidate_actions,
            ltm_id=ltm_id,
            min_traces=min_traces,
            max_traces=max_traces,
            max_antitraces=max_antitraces,
            **params,
        )
        self.perception_config = perception_config
        self.get_logger().info("HardCodedUtilityModel initialized")

    def setup_model(self, trace_length, max_iterations, candidate_actions, ltm_id, train_traces=1, max_traces=50, **params):
        self.episodic_buffer =  TraceBuffer(self, main_size=trace_length, max_traces=max_traces, min_p_traces=train_traces, inputs=['perception'], outputs=[], **params)
        self.learner = DefaultUtilityModelLearner(self, self.episodic_buffer, **params)
        self.confidence_evaluator = None
        self.deliberation = Deliberation(f"{self.name}_deliberation", self, iterations=max_iterations, candidate_actions=candidate_actions, LTM_id=ltm_id, clear_buffer=False, **params)
        self.spin_deliberation()
    
    def predict(self, input_episodes: list[Episode]) -> list[float]:
        "Work in progress"
        distances = []
        i = 0
        for episode in input_episodes:
            # Obtain distances between arms, ball, and box
            left_arm_to_ball = episode.perception["dist_left_arm_ball"][0]["distance"]
            right_arm_to_ball = episode.perception["dist_right_arm_ball"][0]["distance"]
            ball_to_box = episode.perception["dist_ball_box"][0]["distance"]
            ball_in_left_hand = isclose(left_arm_to_ball, 0.0, abs_tol=0.005)
            ball_in_right_hand = isclose(right_arm_to_ball, 0.0, abs_tol=0.005)

            # Calculate angles (denormalize from [0, 1] to [-1, 1] and then to radians)
            left_arm_to_ball_angle_sin = episode.perception["dist_left_arm_ball"][0]["angle_sin"]*2 - 1.0 # Denormalize from [0, 1] to [-1, 1]  
            left_arm_to_ball_angle_cos = episode.perception["dist_left_arm_ball"][0]["angle_cos"]*2 - 1.0 # Denormalize from [0, 1] to [-1, 1]
            left_arm_to_ball_angle_rad = np.arctan2(left_arm_to_ball_angle_sin, left_arm_to_ball_angle_cos)/np.pi # Normalize to [-1, 1] (dividing by pi)
            right_arm_to_ball_angle_sin = episode.perception["dist_right_arm_ball"][0]["angle_sin"]*2 - 1.0 # Denormalize from [0, 1] to [-1, 1]  
            right_arm_to_ball_angle_cos = episode.perception["dist_right_arm_ball"][0]["angle_cos"]*2 - 1.0 # Denormalize from [0, 1] to [-1, 1]
            right_arm_to_ball_angle_rad = np.arctan2(right_arm_to_ball_angle_sin, right_arm_to_ball_angle_cos)/np.pi # Normalize to [-1, 1] (dividing by pi)
            ball_to_box_angle_sin = episode.perception["dist_ball_box"][0]["angle_sin"]*2 - 1.0 # Denormalize from [0, 1] to [-1, 1]
            ball_to_box_angle_cos = episode.perception["dist_ball_box"][0]["angle_cos"]*2 - 1.0 # Denormalize from [0, 1] to [-1, 1]
            ball_to_box_angle_rad = np.arctan2(ball_to_box_angle_sin, ball_to_box_angle_cos)/np.pi # Normalize to [-1, 1] (dividing by pi)
            
            # Position angles
            box_angle = episode.perception["box_angle"][0]["data"]
            ball_angle = episode.perception["ball_angle"][0]["data"]
            
            self.get_logger().debug(f"Episode candidate {i}: \n Left arm to ball distance: {left_arm_to_ball} \n Right arm to ball distance: {right_arm_to_ball} \n Ball to box distance: {ball_to_box} \n Ball in left hand: {ball_in_left_hand} \n Ball in right hand: {ball_in_right_hand} \n Left arm to ball angle (rad): {left_arm_to_ball_angle_rad} \n Right arm to ball angle (rad): {right_arm_to_ball_angle_rad} \n Ball to box angle (rad): {ball_to_box_angle_rad} \n Box angle: {box_angle} \n Ball angle: {ball_angle}")
            i += 1

            # Ball in hand and same side
            if ball_in_left_hand and box_angle < 0.5:
                distances.append(ball_to_box + np.abs(left_arm_to_ball_angle_cos-ball_to_box_angle_cos) + np.abs(left_arm_to_ball_angle_sin-ball_to_box_angle_sin))
                self.get_logger().debug(f"Ball grasped left, same side, distance: {distances[-1]}")
            elif ball_in_right_hand and box_angle > 0.5:
                distances.append(ball_to_box + np.abs(right_arm_to_ball_angle_cos-ball_to_box_angle_cos) + np.abs(right_arm_to_ball_angle_sin-ball_to_box_angle_sin))
                self.get_logger().debug(f"Ball grasped right, same side, distance: {distances[-1]}")
            # Ball in hand but wrong side
            elif ball_in_left_hand:
                distances.append(right_arm_to_ball + np.abs(left_arm_to_ball_angle_rad) + np.abs(right_arm_to_ball_angle_rad) + 5.0) # Penalty for wrong side
                self.get_logger().debug(f"Ball grasped left, wrong side, distance: {distances[-1]}")
            elif ball_in_right_hand:
                distances.append(left_arm_to_ball - np.abs(right_arm_to_ball_angle_rad) + np.abs(left_arm_to_ball_angle_rad) + 5.0) # Penalty for wrong side
                self.get_logger().debug(f"Ball grasped right, wrong side, distance: {distances[-1]}")
            # Ball not in hand
            else:
                if ball_angle < 0.5:
                    distances.append(left_arm_to_ball + np.abs(left_arm_to_ball_angle_rad) + 10.0)
                    self.get_logger().debug(f"Ball not grasped, ball left, distance: {distances[-1]}")
                else:
                    distances.append(right_arm_to_ball + np.abs(right_arm_to_ball_angle_rad) + 10.0)
                    self.get_logger().debug(f"Ball not grasped, ball right, distance: {distances[-1]}")



        # Normalize distances to a range of 0 to 1
        distances = np.array(distances)
        if np.max(distances) == np.min(distances):
            # Avoid division by zero; all values are the same
            normalized_distances = np.ones_like(distances)
        else:
            normalized_distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))

        # Convert normalized distances to utilities (1 - distance)
        utilities = 1 - normalized_distances
        self.get_logger().info(f"Prediction made: {utilities}")
        return utilities

    def execute_callback(self, request, response):
        response = super().execute_callback(request, response)
        return response
    
    def denormalize(self, input_dict, config):
        """
        Denormalize the input dictionary according to the configuration

        :param input_dict: Perception or actuation dictionary.
        :type input_dict: dict
        :param config: Configuration of the perception or actuation bounds.
        :type config: dict
        :return: Denormalized dictionary.
        :rtype: dict
        """        
        out=deepcopy(input_dict)
        for dim in input_dict:
            for param in input_dict[dim][0]:
                config_item = config[dim].get(param, {"type": None})
                if config_item["type"]=="float":
                    bounds=config[dim][param]["bounds"]
                    value=out[dim][0][param]
                    out[dim][0][param]=bounds[0]+(value*(bounds[1]-bounds[0]))
                if config_item["type"] is None:
                    if param == "angle_cos":
                        continue
                    if param == "angle_sin":
                        bounds=config[dim]["angle"]["bounds"]
                        angle_sin = out[dim][0]["angle_sin"]*2 - 1.0  # Denormalize from [0, 1] to [-1, 1]
                        angle_cos = out[dim][0]["angle_cos"]*2 - 1.0  # Denormalize from [0, 1] to [-1, 1]
                        angle_rad = np.arctan2(angle_sin, angle_cos)
                        if bounds == [-180, 180]:
                            angle_deg = angle_rad * (180.0 / np.pi)
                            out[dim][0]["angle"] = angle_deg
                        else:
                            out[dim][0]["angle"] = angle_rad
        return out

class LearnedUtilityModel(UtilityModel):
    def __init__(self, name='utility_model', class_name='cognitive_nodes.utility_model.UtilityModel', prediction_srv_type="cognitive_node_interfaces.srv.PredictUtility", trace_length=20, max_iterations=20, candidate_actions=5, min_traces=5, max_traces=50, max_antitraces=10, ltm_id="", **params):
        super().__init__(
            name=name,
            class_name=class_name,
            prediction_srv_type=prediction_srv_type,
            trace_length=trace_length,
            max_iterations=max_iterations,
            candidate_actions=candidate_actions,
            ltm_id=ltm_id,
            min_traces=min_traces,
            max_traces=max_traces,
            max_antitraces=max_antitraces,
            **params,
        )
        self.min_traces = float(min_traces)
        self.max_traces = max_traces
        self.trace_service = self.create_service(
            AddTrace,
            'utility_model/' + str(name) + '/add_trace',
            self.add_trace_callback,
            callback_group=self.cbgroup_server
        )
        episodes_topic = self.Control["episodes_topic"]
        episodes_msg = self.Control["episodes_msg"]
        self.episode_subscription = self.create_subscription(
            class_from_classname(episodes_msg),
            episodes_topic,
            self.episode_callback,
            0,
            callback_group=self.cbgroup_server
        )

        self.get_logger().info(f"Utility Model created: {self.name}")

    def setup_model(self, trace_length, max_iterations, candidate_actions, ltm_id, min_traces=1, max_traces=50, max_antitraces=10, **params):
        """
        Sets up the Utility Model by initializing the episodic buffer, learner, and confidence evaluator.
        """
        self.episodic_buffer = TraceBuffer(self, main_size=trace_length, max_traces=max_traces, min_traces=min_traces, max_antitraces=max_antitraces, inputs=['perception'], outputs=[], **params)
        self.learner = ANNLearner(self, self.episodic_buffer, **params)
        self.alternative_learner = NoveltyUtilityModelLearner(self, self.episodic_buffer, **params)
        self.confidence_evaluator = DefaultUtilityEvaluator(self, self.learner, self.episodic_buffer, **params)
        self.deliberation = Deliberation(f"{self.name}_deliberation", self, iterations=max_iterations, candidate_actions=candidate_actions, LTM_id=ltm_id, clear_buffer=True, **params)
        self.spin_deliberation()

    def predict(self, input_episodes: list[Episode]) -> list[float]:
        input_data = self.episodic_buffer.buffer_to_matrix(input_episodes, self.episodic_buffer.input_labels)
        predictions = self.learner.call(input_data)
        if predictions is None:
            self.get_logger().warn("Learner not configured, using alternative learner for predictions")
            predictions = self.alternative_learner.call(input_data)
        self.get_logger().info(f"Prediction made: {len(predictions)} episodes")
        self.get_logger().info(f"Predictions: {predictions}")
        return predictions

    def execute_callback(self, request, response):
        response = super().execute_callback(request, response)
        self.get_logger().info(f"Total traces: {self.episodic_buffer.n_traces}, Total antitraces: {self.episodic_buffer.n_antitraces} New traces: {self.episodic_buffer.new_traces}, Min traces: {self.min_traces} {self.episodic_buffer.min_traces}")
        if self.episodic_buffer.new_traces > self.min_traces:
            x_train, y_train = self.episodic_buffer.get_dataset(shuffle=True)
            self.learner.train(x_train, y_train)
            self.episodic_buffer.reset_new_sample_count()
        return response

    def episode_callback(self, msg):
        if not msg.parent_policy: # Parent policy is empty if no specific Utility Model/Policy is being executed
            episode = episode_msg_to_obj(msg)
            linked_goals = self.deliberation.get_linked_goals()
            rewards = [episode.reward_list[goal] for goal in linked_goals if goal in episode.reward_list]
            self.get_logger().info(f"New episode received with rewards: {episode.reward_list}, linked goals: {linked_goals}")
            self.episodic_buffer.add_episode(episode, max(rewards, default=0.0))
            if any(rewards):
                self.get_logger().info(f"New trace added to episodic buffer. Total traces: {self.episodic_buffer.n_traces}, Min traces: {self.episodic_buffer.min_traces}")
                self.deliberation.update_pnodes_reward_basis(episode.old_perception, episode.perception, self.name, episode.reward_list, self.deliberation.LTM_cache)
                if self.episodic_buffer.new_traces > self.min_traces:
                    x_train, y_train = self.episodic_buffer.get_dataset(shuffle=True)
                    self.learner.train(x_train, y_train)
                    self.episodic_buffer.reset_new_sample_count()
        elif msg.parent_policy == "reset_world":
            self.get_logger().info("World reset detected, clearing buffer")
            self.episodic_buffer.clear()

    def add_trace_callback(self, request, response):
        episodes = episode_msg_list_to_obj_list(request.episodes)
        rewards = request.rewards
        for episode, reward in zip(episodes, rewards):
            self.episodic_buffer.add_episode(episode, reward)
        response.added = True
        return response

##### LEARNERS: Place here the Learner classes that implement the learning algorithms for the Utility Model.


class DefaultUtilityModelLearner(Learner):
    """ Default Utility Model class, used when no specific utility model is defined.
    This model does not perform any learning or prediction, it simply returns a constant value.
    """
    def __init__(self, node:UtilityModel, buffer, **params):
        super().__init__(node, buffer, **params)
        self.configured=True


    def train(self):
        return None
    
    def call(self, x):
        output_len = x.shape[0]
        y = np.ones((output_len))
        return y

class NoveltyUtilityModelLearner(Learner):
    """ Default Utility Model class, used when no specific utility model is defined.
    This model provides higher utility to states not visited previously.
    """
    def __init__(self, node:UtilityModel, buffer, **params):
        super().__init__(node, buffer, **params)
        self.configured=True


    def train(self):
        return None

    def call(self, x):
        previous_episodes, _ = self.buffer.get_train_samples()
        # Compute novelty based on previous episodes
        novelty = self.compute_novelty(previous_episodes, x)
        return novelty
    
    def compute_novelty(self, previous_episodes, candidate_episodes):
        # previous_episodes: (N, D), candidate_episodes: (M, D)
        if previous_episodes is None or len(previous_episodes) == 0:
            # If no previous episodes, all candidates are maximally novel
            return np.ones(candidate_episodes.shape[0])
        # Compute pairwise distances (M, N)
        dists = np.linalg.norm(candidate_episodes[:, None, :] - previous_episodes[None, :, :], axis=2)
        # For each candidate, get the minimum distance to any previous episode
        min_dists = np.min(dists, axis=1)
        # Normalize to 0-1
        if np.max(min_dists) == np.min(min_dists):
            # Avoid division by zero; all values are the same
            return np.ones_like(min_dists)
        normalized = (min_dists - np.min(min_dists)) / (np.max(min_dists) - np.min(min_dists))
        return normalized

##### EVALUATORS: Place here the Evaluator classes that implement the evaluation algorithms for the Utility Model.

class DefaultUtilityEvaluator(Evaluator):
    """ Default Utility Evaluator class, used when no specific utility evaluator is defined.
    This evaluator does not perform any evaluation, it simply returns a constant value.
    """
    def __init__(self, node: UtilityModel, learner:DefaultUtilityModelLearner, buffer: None, **params):
        super().__init__(node, learner, buffer, **params)
        self.model_confidence = 1.0

    def evaluate(self):
        """
        Evaluates the input episodes and returns a list of evaluated episodes.

        :param input_episodes: List of input episodes to evaluate.
        :type input_episodes: list
        :return: List of evaluated episodes.
        :rtype: list
        """
        return self.model_confidence


def main(args=None):
    rclpy.init(args=args)

    utility_model = UtilityModel()

    rclpy.spin(utility_model)

    utility_model.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()