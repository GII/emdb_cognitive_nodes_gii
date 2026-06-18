import rclpy
import threading
import numpy as np
from copy import deepcopy
from rclpy.node import Node
from math import isclose
from rclpy.executors import SingleThreadedExecutor

from core.utils import class_from_classname
from cognitive_nodes.episode import Episode, episode_msg_to_obj, episode_obj_to_msg, episode_msg_list_to_obj_list
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
        """Initialize the Utility Model with deliberation and learning capabilities.

        :param name: The name of the Utility Model instance, defaults to 'utility_model'
        :type name: str, optional
        :param class_name: The fully qualified class name for the Utility Model, defaults to 'cognitive_nodes.utility_model.UtilityModel'
        :type class_name: str, optional
        :param prediction_srv_type: The service type for predictions, defaults to "cognitive_node_interfaces.srv.PredictUtility"
        :type prediction_srv_type: str, optional
        :param trace_length: Maximum number of traces to store in the episodic buffer, defaults to 20
        :type trace_length: int, optional
        :param max_iterations: Maximum number of iterations for the deliberation process, defaults to 20
        :type max_iterations: int, optional
        :param candidate_actions: Number of candidate actions to generate during deliberation, defaults to 5
        :type candidate_actions: int, optional
        :param ltm_id: Identifier for the Long-Term Memory cache used in deliberation, defaults to ""
        :type ltm_id: str, optional
        :param params: Additional keyword parameters reserved for future use.
        :type params: dict
        """        
        super().__init__(name, class_name, prediction_srv_type=prediction_srv_type, node_type="utility_model", **params)
        self.configure_activation_inputs(self.neighbors)
        self.setup_model(trace_length=trace_length, max_iterations=max_iterations, candidate_actions=candidate_actions, ltm_id=ltm_id, **params)
        self.execute_service = self.create_service(
            Execute,
            'utility_model/' + str(name) + '/execute',
            self.execute_callback,
            callback_group=self.cbgroup_server
        )

    def setup_model(self, trace_length, max_iterations, candidate_actions, ltm_id, **params):
        """Sets up the Utility Model by initializing the episodic buffer, learner, and confidence evaluator.

        :param trace_length: Maximum number of traces to store in the trace buffer.
        :type trace_length: int
        :param max_iterations: Maximum number of iterations for the deliberation process.
        :type max_iterations: int
        :param candidate_actions: Number of candidate actions to generate during deliberation.
        :type candidate_actions: int
        :param ltm_id: Identifier for the Long-Term Memory cache used in deliberation.
        :type ltm_id: str
        :param params: Additional keyword parameters reserved for future use.
        :type params: dict
        """        
        self.episodic_buffer = TraceBuffer(self, main_size=trace_length, inputs=['perception'], outputs=[], **params)
        self.learner = DefaultUtilityModelLearner(self, self.episodic_buffer, **params)
        self.confidence_evaluator = DefaultUtilityEvaluator(self, self.learner, self.episodic_buffer, **params)
        self.deliberation = Deliberation(f"{self.name}_deliberation", self, iterations=max_iterations, candidate_actions=candidate_actions, LTM_id=ltm_id, clear_buffer=True, **params)
        self.spin_deliberation()

    def spin_deliberation(self):
        """Starts a separate thread to spin the deliberation executor.
        """        
        self.deliberation_executor = SingleThreadedExecutor()
        self.deliberation_executor.add_node(self.deliberation)
        self.deliberation_thread = threading.Thread(target=self.deliberation_executor.spin)
        self.deliberation_thread.start()

    def calculate_activation(self, perception = None, activation_list=None):
        """Calculate the activation level of the utility model based on perception or activation inputs.

        :param perception: Perception data to use for activation calculation, defaults to None
        :type perception: dict, optional
        :param activation_list: List of activations from connected nodes to aggregate, defaults to None
        :type activation_list: list, optional
        :return: None
        :rtype: None
        """        
        if activation_list and self.learner.configured:
            self.calculate_activation_max(activation_list)
        else:
            self.activation.activation=0.0
            self.activation.timestamp=self.get_clock().now().to_msg()

    def predict(self, input_episodes: list[Episode]) -> list[float]:
        """Predict the expected utilities for a list of input episodes using the learner model.

        :param input_episodes: List of Episode objects to predict utilities for.
        :type input_episodes: list[Episode]
        :return: List of predicted utility values, one for each input episode.
        :rtype: list[float]
        """        
        input_data = self.episodic_buffer.buffer_to_matrix(input_episodes, self.episodic_buffer.input_labels)
        expected_utilities = self.learner.call(input_data)
        self.get_logger().info(f"Predictions: {expected_utilities}")
        return expected_utilities
    
    def predict_callback(self, request, response):
        self.get_logger().info("Predicting ...")

        try:
            input_episodes = episode_msg_list_to_obj_list(request.episodes)
            expected_utilities = self.predict(input_episodes)
            expected_utilities = [float(x) for x in expected_utilities]

            response.expected_utilities = expected_utilities
            response.valid = True

            self.get_logger().info(
                f"Prediction made: {len(input_episodes)} episodes"
            )
            self.get_logger().info(
                f"Predictions: {expected_utilities}"
            )

        except Exception as exc:
            self.get_logger().error(f"Prediction failed: {exc}")
            response.expected_utilities = []
            response.valid = False

        return response
    
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
        response.episode = episode_obj_to_msg(self.deliberation.summary_episode)

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
        """Sets up the Novelty Utility Model by initializing the episodic buffer, learner, and deliberation process.

        :param trace_length: Maximum number of traces to store in the episodic buffer.
        :type trace_length: int
        :param max_iterations: Maximum number of iterations for the deliberation process.
        :type max_iterations: int
        :param candidate_actions: Number of candidate actions to generate during deliberation.
        :type candidate_actions: int
        :param ltm_id: Identifier for the Long-Term Memory cache used in deliberation.
        :type ltm_id: str
        :param train_traces: Minimum number of positive traces required in the buffer, defaults to 1
        :type train_traces: int, optional
        :param max_traces: Maximum number of traces to store in the buffer, defaults to 50
        :type max_traces: int, optional
        """        
        self.episodic_buffer = EpisodicBuffer(self, main_size=trace_length, secondary_size=0, train_split=1.0, inputs=['perception'], outputs=[], **params)
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
        """Sets up the Hard Coded Utility Model by initializing the episodic buffer, learner, and deliberation process.

        :param trace_length: Maximum number of traces to store in the trace buffer.
        :type trace_length: int
        :param max_iterations: Maximum number of iterations for the deliberation process.
        :type max_iterations: int
        :param candidate_actions: Number of candidate actions to generate during deliberation.
        :type candidate_actions: int
        :param ltm_id: Identifier for the Long-Term Memory cache used in deliberation.
        :type ltm_id: str
        :param train_traces: Minimum number of positive traces required in the buffer, defaults to 1
        :type train_traces: int, optional
        :param max_traces: Maximum number of traces to store in the buffer, defaults to 50
        :type max_traces: int, optional
        :param params: Additional keyword parameters reserved for future use.
        :type params: dict
        """        
        self.episodic_buffer =  TraceBuffer(self, main_size=trace_length, max_traces=max_traces, min_p_traces=train_traces, inputs=['perception'], outputs=[], **params)
        self.learner = DefaultUtilityModelLearner(self, self.episodic_buffer, **params)
        self.confidence_evaluator = None
        self.deliberation = Deliberation(f"{self.name}_deliberation", self, iterations=max_iterations, candidate_actions=candidate_actions, LTM_id=ltm_id, clear_buffer=False, **params)
        self.spin_deliberation()
    
    def predict(self, input_episodes: list[Episode]) -> list[float]:
        """Predict utilities for input episodes using hard-coded heuristics for ball-to-box task.
        
        Computes utilities based on distances and angles between robot arms, ball, and target box,
        with penalties for suboptimal configurations and rewards for goal-oriented states.

        :param input_episodes: List of Episode objects containing perception data to evaluate.
        :type input_episodes: list[Episode]
        :return: List of normalized utility values (0-1 range) for each input episode.
        :rtype: list[float]
        """        
        distances = []
        i = 0
        for episode in input_episodes:
            # Obtain distances between arms, ball, and box
            left_arm_to_ball = episode.perception["dist_left_arm_ball"][0]["distance"]
            right_arm_to_ball = episode.perception["dist_right_arm_ball"][0]["distance"]
            ball_to_box = episode.perception["dist_ball_box"][0]["distance"]
            ball_in_left_hand = isclose(left_arm_to_ball, 0.0, abs_tol=0.025)
            ball_in_right_hand = isclose(right_arm_to_ball, 0.0, abs_tol=0.025)

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
    """
    Learned Utility Model class: A utility model that learns from episodic traces to predict utilities.
    """    
    def __init__(
        self,
        name="utility_model",
        class_name="cognitive_nodes.utility_model.UtilityModel",
        prediction_srv_type="cognitive_node_interfaces.srv.PredictUtility",
        trace_length=20,
        max_iterations=20,
        candidate_actions=5,
        min_traces=5,
        max_traces=50,
        max_antitraces=10,
        train_traces=5,
        train_every=1,
        validation_split=0.1,
        reward_factor=1.0,
        ltm_id="",
        **params,
    ):
        """Initialize the Learned Utility Model with machine learning capabilities.

        :param name: The name of the Utility Model instance, defaults to "utility_model"
        :type name: str, optional
        :param class_name: The fully qualified class name for the Utility Model, defaults to "cognitive_nodes.utility_model.UtilityModel"
        :type class_name: str, optional
        :param prediction_srv_type: The service type for predictions, defaults to "cognitive_node_interfaces.srv.PredictUtility"
        :type prediction_srv_type: str, optional
        :param trace_length: Maximum number of traces to store in the episodic buffer, defaults to 20
        :type trace_length: int, optional
        :param max_iterations: Maximum number of iterations for the deliberation process, defaults to 20
        :type max_iterations: int, optional
        :param candidate_actions: Number of candidate actions to generate during deliberation, defaults to 5
        :type candidate_actions: int, optional
        :param min_traces: Minimum number of traces required before training can begin, defaults to 5
        :type min_traces: int, optional
        :param max_traces: Maximum number of traces to store in the buffer, defaults to 50
        :type max_traces: int, optional
        :param max_antitraces: Maximum number of antitraces (negative examples) to store, defaults to 10
        :type max_antitraces: int, optional
        :param train_traces: Number of new traces required to trigger a training step, defaults to 5
        :type train_traces: int, optional
        :param train_every: Frequency of training updates (number of new traces before training), defaults to 1
        :type train_every: int, optional
        :param validation_split: Fraction of data to use for validation during training, defaults to 0.1
        :type validation_split: float, optional
        :param reward_factor: Scaling factor applied to rewards, defaults to 1.0
        :type reward_factor: float, optional
        :param ltm_id: Identifier for the Long-Term Memory cache used in deliberation, defaults to ""
        :type ltm_id: str, optional
        :param params: Additional keyword parameters reserved for future use.
        :type params: dict
        """        
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
            reward_factor=reward_factor,
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
        self.train_every = train_every
        self.train_traces = train_traces
        self.validation_split = validation_split
        self.update_semaphore = threading.Semaphore()

        self.get_logger().info(f"Utility Model created: {self.name}")

    def calculate_activation(self, perception = None, activation_list=None):
        """Calculate the activation level of the utility model based on perception or activation inputs.

        :param perception: Perception data to use for activation calculation, defaults to None
        :type perception: dict, optional
        :param activation_list: List of activations from connected nodes to aggregate, defaults to None
        :type activation_list: list, optional
        :return: None
        :rtype: None
        """
        if self.update_semaphore.acquire(blocking=False):        
            if activation_list and self.learner.configured:
                self.calculate_activation_max(activation_list)
            else:
                self.activation.activation=0.0
                self.activation.timestamp=self.get_clock().now().to_msg()
            self.update_semaphore.release()

    def setup_model(self, trace_length, max_iterations, candidate_actions, ltm_id, min_traces=1, max_traces=50, max_antitraces=10, **params):
        """Sets up the Learned Utility Model by initializing the episodic buffer, learner, and confidence evaluator.

        :param trace_length: Maximum number of traces to store in the trace buffer.
        :type trace_length: int
        :param max_iterations: Maximum number of iterations for the deliberation process.
        :type max_iterations: int
        :param candidate_actions: Number of candidate actions to generate during deliberation.
        :type candidate_actions: int
        :param ltm_id: Identifier for the Long-Term Memory cache used in deliberation.
        :type ltm_id: str
        :param min_traces: Minimum number of traces required before training can begin, defaults to 1
        :type min_traces: int, optional
        :param max_traces: Maximum number of traces to store in the buffer, defaults to 50
        :type max_traces: int, optional
        :param max_antitraces: Maximum number of antitraces (negative examples) to store, defaults to 10
        :type max_antitraces: int, optional
        :param params: Additional keyword parameters reserved for future use.
        :type params: dict
        """        
        self.episodic_buffer = TraceBuffer(self, main_size=trace_length, max_traces=max_traces, min_traces=min_traces, max_antitraces=max_antitraces, inputs=['perception'], outputs=[], **params)
        self.learner = ANNLearner(self, self.episodic_buffer, **params)
        self.alternative_learner = DefaultUtilityModelLearner(self, self.episodic_buffer, **params)
        self.confidence_evaluator = DefaultUtilityEvaluator(self, self.learner, self.episodic_buffer, **params)
        self.deliberation = Deliberation(f"{self.name}_deliberation", self, iterations=max_iterations, candidate_actions=candidate_actions, LTM_id=ltm_id, clear_buffer=True, **params)
        self.spin_deliberation()

    def predict(self, input_episodes: list[Episode]) -> list[float]:
        """Predict utilities for input episodes using the learner model.

        :param input_episodes: List of Episode objects to predict utilities for.
        :type input_episodes: list[Episode]
        :return: List of predicted utility values for each input episode.
        :rtype: list[float]
        """        
        input_data = self.episodic_buffer.buffer_to_matrix(input_episodes, self.episodic_buffer.input_labels)
        predictions = self.learner.call(input_data)
        if predictions is None:
            self.get_logger().warn("Learner not configured, using alternative learner for predictions")
            predictions = self.alternative_learner.call(input_data)
        self.get_logger().info(f"Prediction made: {len(predictions)} episodes")
        self.get_logger().info(f"Predictions: {predictions}")
        return predictions

    def train_step(self):
        """Perform a training step for the utility model if sufficient new traces are available.
        """        
        if self.episodic_buffer.n_traces >= self.min_traces and self.episodic_buffer.new_traces >= self.train_every:
            sample_size = max(self.train_traces, self.episodic_buffer.new_traces)
            self.get_logger().info(f"Training Utility Model with {sample_size} new traces")
            x_train, y_train = self.episodic_buffer.get_dataset(shuffle=True, n_samples=sample_size)
            self.learner.train(x_train, y_train, validation_split=self.validation_split)
            self.episodic_buffer.reset_new_sample_count()

    def execute_callback(self, request, response):
        """Execute the deliberation process and perform a training step if sufficient traces are available.
        
        This method extends the parent class execution callback by adding a training step
        after deliberation completes. It logs the current state of the episodic buffer
        including trace counts and triggers model training if the minimum trace threshold is met.

        :param request: The request object from the Execute service containing execution parameters.
        :type request: cognitive_node_interfaces.srv.Execute.Request
        :param response: The response object to be populated with execution results.
        :type response: cognitive_node_interfaces.srv.Execute.Response
        :return: The response object containing the policy name and summary episode from deliberation.
        :rtype: cognitive_node_interfaces.srv.Execute.Response
        """       
        response = super().execute_callback(request, response)
        self.get_logger().info(f"Total traces: {self.episodic_buffer.n_traces}, Total antitraces: {self.episodic_buffer.n_antitraces} New traces: {self.episodic_buffer.new_traces}, Min traces: {self.min_traces} {self.episodic_buffer.min_traces}")
        self.train_step()
        return response

    def episode_callback(self, msg):
        """Handle incoming episode messages and update the episodic buffer and learning model.

        This callback processes episode messages from other policies. It handles world resets by clearing
        the buffer and processes new episodes by extracting rewards for linked goals, adding them to the
        episodic buffer, and triggering model training if positive rewards are present. Long-term memory
        is asynchronously updated in a separate thread when new successful traces are added.

        :param msg: The episode message received from the subscription, containing perception data,
                    rewards, and parent policy information.
        :type msg: cognitive_node_interfaces.msg.Episode
        """        
        if msg.parent_policy == "reset_world":
            self.get_logger().info("World reset detected, clearing buffer")
            self.episodic_buffer.clear()
        elif msg.parent_policy!=self.name: # Filter self generated episodes as those are handled by the deliberation process
            episode = episode_msg_to_obj(msg)
            linked_goals = self.deliberation.get_linked_goals()
            rewards = [episode.reward_list[goal] for goal in linked_goals if goal in episode.reward_list]
            self.get_logger().info(f"New episode received with rewards: {episode.reward_list}, linked goals: {linked_goals}")
            self.episodic_buffer.add_episode(episode, max(rewards, default=0.0))
            if any(rewards):
                self.get_logger().info(f"New trace added to episodic buffer. Total traces: {self.episodic_buffer.n_traces}, Min traces: {self.episodic_buffer.min_traces}")
                ltm_update_thread = threading.Thread(target=self.update_ltm_and_train, args=(episode.old_perception, episode.perception, self.name, episode.reward_list, self.deliberation.LTM_cache))
                ltm_update_thread.start()
                

    def update_ltm_and_train(self, old_perception, perception, policy, reward_list, ltm_cache):
        """Update the Long-Term Memory cache with new reward basis information.
        This method updates the Long-Term Memory (LTM) cache with new reward basis information
        based on the provided perceptions, policy, and reward list. It uses a semaphore to ensure
        thread-safe access to the LTM during the update process.
        :param old_perception: The previous perception state before the episode.
        :type old_perception: dict
        :param perception: The current perception state after the episode.
        :type perception: dict
        :param policy: The policy name associated with the episode.
        :type policy: str
        :param reward_list: The list of rewards associated with the episode.
        :type reward_list: dict
        :param ltm_cache: The Long-Term Memory cache to be updated.
        :type ltm_cache: dict
        """
        self.update_semaphore.acquire()
        self.deliberation.update_pnodes_reward_basis(old_perception, perception, policy, reward_list, ltm_cache)
        self.train_step()
        self.update_semaphore.release()
        

    def add_trace_callback(self, request, response):
        """Service callback to add traces to the episodic buffer.
        :param request: The request object containing episodes and rewards to be added.
        :type request: cognitive_node_interfaces.srv.AddTrace.Request
        :param response: The response object to be populated with the result of the addition.
        :type response: cognitive_node_interfaces.srv.AddTrace.Response
        :return: The response object indicating whether the traces were successfully added.
        :rtype: cognitive_node_interfaces.srv.AddTrace.Response
        """
        episodes = episode_msg_list_to_obj_list(request.episodes)
        rewards = request.rewards
        for episode, reward in zip(episodes, rewards):
            self.episodic_buffer.add_episode(episode, reward)
        response.added = True
        return response


class DummyUtilityModel(LearnedUtilityModel):
    """
    Dummy Utility Model class
    This model is used as a placeholder and does not perform any actual utility computation.
    It inherits from the UtilityModel class.
    """
    def __init__(self, name='dummy_utility_model', class_name='cognitive_nodes.utility_model.UtilityModel', prediction_srv_type="cognitive_node_interfaces.srv.PredictUtility", trace_length=20, max_iterations=20, candidate_actions=5, ltm_id="", **params):
        super().__init__(name=name, class_name=class_name, prediction_srv_type=prediction_srv_type, trace_length=trace_length, max_iterations=max_iterations, candidate_actions=candidate_actions, ltm_id=ltm_id, **params)

    def calculate_activation(self, perception=None, activation_list=None):
        self.activation.activation = 0.0
        self.activation.timestamp = self.get_clock().now().to_msg()
        return self.activation

class QUtilityModel(LearnedUtilityModel):
    """
    Q-Learning Utility Model class: A utility model that uses Q-learning to predict utilities.
    
    This model implements a Q-learning approach with a target network for stable training.
    It maintains two neural networks: a main learner and a target learner that are 
    periodically synchronized. The model predicts Q-values for state-action pairs and 
    uses these to compute optimal utilities during deliberation.
    """
    def __init__(
            self, 
            name="utility_model", 
            class_name="cognitive_nodes.utility_model.UtilityModel", 
            prediction_srv_type="cognitive_node_interfaces.srv.PredictUtility", 
            trace_length=20, 
            max_iterations=20, 
            candidate_actions=5, 
            min_traces=5, 
            max_traces=50, 
            max_antitraces=10, 
            train_traces=5, 
            train_every=1,
            replace_every=5,
            discount_factor=0.90,
            validation_split=0.1, 
            reward_factor=1.0,
            ltm_id="", 
            **params
            ):
        """Initialize the Q-Learning Utility Model with target network.

        :param name: The name of the Utility Model instance, defaults to "utility_model"
        :type name: str, optional
        :param class_name: The fully qualified class name for the Utility Model, defaults to "cognitive_nodes.utility_model.UtilityModel"
        :type class_name: str, optional
        :param prediction_srv_type: The service type for predictions, defaults to "cognitive_node_interfaces.srv.PredictUtility"
        :type prediction_srv_type: str, optional
        :param trace_length: Maximum number of traces to store in the episodic buffer, defaults to 20
        :type trace_length: int, optional
        :param max_iterations: Maximum number of iterations for the deliberation process, defaults to 20
        :type max_iterations: int, optional
        :param candidate_actions: Number of candidate actions to generate during deliberation, defaults to 5
        :type candidate_actions: int, optional
        :param min_traces: Minimum number of traces required before training can begin, defaults to 5
        :type min_traces: int, optional
        :param max_traces: Maximum number of traces to store in the buffer, defaults to 50
        :type max_traces: int, optional
        :param max_antitraces: Maximum number of antitraces (negative examples) to store, defaults to 10
        :type max_antitraces: int, optional
        :param train_traces: Number of new traces required to trigger a training step, defaults to 5
        :type train_traces: int, optional
        :param train_every: Frequency of training updates (number of new traces before training), defaults to 1
        :type train_every: int, optional
        :param replace_every: Number of training steps before updating target network, defaults to 5
        :type replace_every: int, optional
        :param discount_factor: Discount factor for future rewards in Q-value calculation, defaults to 0.90
        :type discount_factor: float, optional
        :param validation_split: Fraction of data to use for validation during training, defaults to 0.1
        :type validation_split: float, optional
        :param reward_factor: Scaling factor applied to rewards, defaults to 1.0
        :type reward_factor: float, optional
        :param ltm_id: Identifier for the Long-Term Memory cache used in deliberation, defaults to ""
        :type ltm_id: str, optional
        :param params: Additional keyword parameters reserved for future use.
        :type params: dict
        """        
        super().__init__(
            name=name, 
            class_name=class_name, 
            prediction_srv_type=prediction_srv_type, 
            trace_length=trace_length, 
            max_iterations=max_iterations, 
            candidate_actions=candidate_actions, 
            min_traces=min_traces, 
            max_traces=max_traces, 
            max_antitraces=max_antitraces, 
            train_traces=train_traces, 
            train_every=train_every,
            validation_split=validation_split, 
            reward_factor=reward_factor,
            ltm_id=ltm_id, 
            **params
            )
        
        self.candidate_actions = candidate_actions
        self.replace_every = replace_every
        self.discount_factor = discount_factor
        self.train_step_count = 0


    def setup_model(self, trace_length, max_iterations, candidate_actions, ltm_id, min_traces=1, max_traces=50, max_antitraces=10, **params):
        """Sets up the Q-Learning Utility Model by initializing dual networks and episodic buffer.
        
        :param trace_length: Maximum number of traces to store in the trace buffer.
        :type trace_length: int
        :param max_iterations: Maximum number of iterations for the deliberation process.
        :type max_iterations: int
        :param candidate_actions: Number of candidate actions to generate during deliberation.
        :type candidate_actions: int
        :param ltm_id: Identifier for the Long-Term Memory cache used in deliberation.
        :type ltm_id: str
        :param min_traces: Minimum number of traces required before training can begin, defaults to 1
        :type min_traces: int, optional
        :param max_traces: Maximum number of traces to store in the buffer, defaults to 50
        :type max_traces: int, optional
        :param max_antitraces: Maximum number of antitraces (negative examples) to store, defaults to 10
        :type max_antitraces: int, optional
        :param params: Additional keyword parameters reserved for future use.
        :type params: dict
        """
        self.episodic_buffer = TraceBuffer(self, main_size=trace_length, max_traces=max_traces, min_traces=min_traces, max_antitraces=max_antitraces, inputs=['perception'], outputs=[], **params)
        self.learner = ANNLearner(self, self.episodic_buffer, **params)
        self.target_learner = ANNLearner(self, self.episodic_buffer, **params)
        self.alternative_learner = DefaultUtilityModelLearner(self, self.episodic_buffer, **params)
        self.confidence_evaluator = DefaultUtilityEvaluator(self, self.learner, self.episodic_buffer, **params)
        self.deliberation = Deliberation(f"{self.name}_deliberation", self, iterations=max_iterations, candidate_actions=candidate_actions, LTM_id=ltm_id, clear_buffer=True, **params)
        self.spin_deliberation()
    
    def train_step(self):
        """Perform a training step using Q-learning with target network updates.
        
        Initializes both the main and target learners on first training. Computes Q-values
        for next states using the target network and updates the main learner with computed
        Q-targets. Periodically synchronizes the target network with the main learner weights.
        """
        if self.episodic_buffer.n_traces >= self.min_traces and self.episodic_buffer.new_traces >= self.train_every:
            # Initialize both learners if not configured
            if not self.learner.configured:
                x_train, rewards = self.episodic_buffer.get_dataset()
                rewards = rewards.reshape(-1, 1)
                self.learner.configure_model(x_train.shape[1], rewards.shape[1])
                self.target_learner.configure_model(x_train.shape[1], rewards.shape[1])
                weights = self.learner.get_weights()
                self.target_learner.set_weights(weights)

            # Train every fixed number of episodes when buffer is full
            sample_size = max(self.train_traces, self.episodic_buffer.new_traces)
            self.get_logger().info(
                f"Training on {sample_size} new traces. Total traces: {self.episodic_buffer.n_traces}"
            )
            episodes, rewards = self.episodic_buffer.get_flattened_traces(n_samples=sample_size)
            x_train = self.episodic_buffer.buffer_to_matrix(episodes, self.episodic_buffer.input_labels)

            # Compute target Q-values
            candidate_states_actions = self.generate_candidates_matrix(episodes)
            if self.deliberation.current_world is None:
                self.deliberation.current_world = self.deliberation.get_linked_world_model()[0]
            next_states = self.deliberation.predict_perceptions(self.deliberation.current_world, candidate_states_actions)
            q_values_next = self.predict(next_states, target_learner=True)
            max_q_values_next = self.obtain_maximum_value(q_values_next)
            y_train = np.array([reward + self.discount_factor * max_q if reward < 1.0 else reward for reward, max_q in zip(rewards, max_q_values_next)])

            # Train the learner with the computed Q-values
            x_train, y_train = self.episodic_buffer._shuffle_dataset(x_train, y_train)
            self.learner.train(x_train, y_train, validation_split=self.validation_split, verbose=1)
            self.episodic_buffer.reset_new_sample_count()
            self.train_step_count += 1
            
            # Update target network every fixed number of training steps
            if self.train_step_count >= self.replace_every:
                self.get_logger().info(
                    f"Updating target network after {self.train_step_count} training steps."
                )
                weights = self.learner.get_weights()
                self.target_learner.set_weights(weights)
                self.train_step_count = 0

    def predict(self, input_episodes: list[Episode], target_learner=False) -> list[float]:
        """Predict Q-values for input episodes using either main or target learner.
        
        :param input_episodes: List of Episode objects to predict Q-values for.
        :type input_episodes: list[Episode]
        :param target_learner: If True, use target network for predictions; otherwise use main learner, defaults to False
        :type target_learner: bool, optional
        :return: List of predicted Q-values for each input episode.
        :rtype: list[float]
        """
        input_data = self.episodic_buffer.buffer_to_matrix(input_episodes, self.episodic_buffer.input_labels)
        learner = self.target_learner if target_learner else self.learner
        predictions = learner.call(input_data)
        if predictions is None:
            self.get_logger().warn("Learner not configured, using alternative learner for predictions")
            predictions = self.alternative_learner.call(input_data)
        self.get_logger().info(f"Prediction made: {len(predictions)} episodes")
        self.get_logger().info(f"Predictions: {predictions}")
        return predictions


    def generate_candidates_matrix(self, buffer, algorithm="latin"):
        """Generate candidate action matrices from episode buffer states.
        
        For each episode in the buffer, generates multiple candidate actions and creates
        composite state-action pairs for evaluation.

        :param buffer: List of Episode objects representing states.
        :type buffer: list[Episode]
        :param algorithm: Algorithm for candidate action generation, defaults to "latin"
        :type algorithm: str, optional
        :return: List of candidate episodes with state-action combinations.
        :rtype: list[Episode]
        """
        candidate_episodes = []
        for episode in buffer:
            candidates = self.deliberation.generate_candidate_actions(episode.perception, algorithm=algorithm)
            candidate_episodes.extend(candidates)
        return candidate_episodes

    def obtain_maximum_value(self, q_values):
        """Extract maximum Q-value for each state from flattened candidate Q-values.
        
        Given Q-values computed for all candidate actions, returns the maximum Q-value
        for each state by reshaping and reducing across the candidate actions dimension.

        :param q_values: Flattened array of Q-values of shape (n_states * candidate_actions,)
        :type q_values: np.ndarray
        :return: Array of maximum Q-values for each state of shape (n_states,)
        :rtype: np.ndarray
        :raises ValueError: If Q-values length is not divisible by candidate_actions
        """
        q_values = np.asarray(q_values).flatten()
        
        # Reshape to (n_states, candidate_actions) and take max along candidate_actions axis
        n_total_rows = q_values.shape[0]
        n_states = n_total_rows // self.candidate_actions
        
        if n_total_rows % self.candidate_actions != 0:
            raise ValueError(f"Q-values length {n_total_rows} is not divisible by candidate_actions {self.candidate_actions}")
        
        # Reshape and take maximum along the candidate_actions dimension
        q_values_reshaped = q_values.reshape(n_states, self.candidate_actions)
        max_q_values = np.max(q_values_reshaped, axis=1)
    
        return max_q_values


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
        """Returns a vector of ones as the utility

        :param x: Input data for prediction.
        :type x: np.ndarray
        :return: Model predictions as a numpy array.
        :rtype: np.ndarray
        """        
        output_len = x.shape[0]
        y = np.ones((output_len))
        return y

class NoveltyUtilityModelLearner(Learner):
    """ Novelty Utility Model class, used as an exploration method.
    This model provides higher utility to states not visited previously.
    """
    def __init__(self, node:UtilityModel, buffer, **params):
        super().__init__(node, buffer, **params)
        self.configured=True


    def train(self):
        return None

    def call(self, x):
        """Returns the computed novelty for the input data

        :param x: Input data for prediction.
        :type x: np.ndarray
        :return: Model predictions as a numpy array.
        :rtype: np.ndarray
        """   
        previous_episodes, _ = self.buffer.get_train_samples()
        # Compute novelty based on previous episodes
        novelty = self.compute_novelty(previous_episodes, x)
        return novelty
    
    def compute_novelty(self, previous_episodes, candidate_episodes):
        """
        Compute normalized novelty scores for candidate episodes relative to previously seen episodes.

        :param previous_episodes: Array of prior episode embeddings with shape (N, D); if empty or None, all candidates are maximally novel.
        :type previous_episodes: np.ndarray | None
        :param candidate_episodes: Array of candidate episode embeddings with shape (M, D).
        :type candidate_episodes: np.ndarray
        :return: Novelty scores in [0, 1] for each candidate, where higher values indicate greater novelty.
        :rtype: np.ndarray
        """
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
