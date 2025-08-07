import rclpy
import numpy as np
from rclpy.node import Node

from cognitive_nodes.episode import Episode
from cognitive_nodes.episodic_buffer import EpisodicBuffer
from cognitive_processes.deliberation import Deliberation

from cognitive_nodes.deliberative_model import DeliberativeModel, Learner, Evaluator
from cognitive_node_interfaces.srv import Execute


class UtilityModel(DeliberativeModel):
    """
    Utility Model class
    """
    def __init__(self, name='utility_model', class_name = 'cognitive_nodes.utility_model.UtilityModel', max_iterations=20, candidate_actions = 5, ltm_id="", **params):
        """
        Constructor of the Utility Model class.

        Initializes a Utility Model instance with the given name and registers it in the LTM.

        :param name: The name of the Utility Model instance.
        :type name: str
        :param class_name: The name of the Utility Model class.
        :type class_name: str
        """
        super().__init__(name, class_name, prediction_srv_type="cognitive_node_interfaces.srv.PredictUtility", **params)
        self.setup_model(max_iterations=max_iterations, candidate_actions=candidate_actions, ltm_id=ltm_id, **params)
        self.execute_service = self.create_service(
            Execute,
            'utility_model/' + str(name) + '/execute',
            self.execute_callback,
            callback_group=self.cbgroup_server
        )

    def setup_model(self, max_iterations, candidate_actions, ltm_id, **params):
        """
        Sets up the Utility Model by initializing the episodic buffer, learner, and confidence evaluator.
        """
        self.episodic_buffer = EpisodicBuffer(self, main_size=200, secondary_size=50, inputs=['perception'], outputs=[])
        self.learner = DefaultUtilityModelLearner(self.episodic_buffer, **params)
        self.confidence_evaluator = DefaultUtilityEvaluator(self, self.learner, self.episodic_buffer, **params)
        self.deliberation = Deliberation(self, iterations=max_iterations, candidate_actions=candidate_actions, LTM_id=ltm_id, clear_buffer=True, **params)


    def calculate_activation(self, perception = None, activation_list=None):
        """
        Returns the the activation value of the Utility Model.
        Dummy, for the moment, as it returns a random value.

        :param perception: The given perception.
        :type perception: dict
        :return: The activation of the instance.
        :rtype: float
        """
        if activation_list:
            self.calculate_activation_max(activation_list)
        else:
            self.activation.activation=0.0
            self.activation.timestamp=self.get_clock().now().to_msg()

    def predict(self, input_episodes: list[Episode]) -> list[float]:
            input_data = self.episodic_buffer.buffer_to_matrix(input_episodes, self.episodic_buffer.input_labels)
            expected_utilities = self.learner.predict(input_data)
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
    def __init__(self, name='utility_model', class_name='cognitive_nodes.utility_model.UtilityModel', max_iterations=20, candidate_actions=5, ltm_id="", **params):
        super().__init__(name, class_name, max_iterations, candidate_actions, ltm_id, **params)
    
    def setup_model(self, max_iterations, candidate_actions, ltm_id, **params):
        self.episodic_buffer = EpisodicBuffer(self, main_size=40, secondary_size=0, inputs=['perception'], outputs=[])
        self.learner = NoveltyUtilityModelLearner(self, self.episodic_buffer, **params)
        self.confidence_evaluator = DefaultUtilityEvaluator(self, self.learner, self.episodic_buffer, **params)
        self.deliberation = Deliberation(self, iterations=max_iterations, candidate_actions=candidate_actions, LTM_id=ltm_id, clear_buffer=False, **params)

class HardCodedUtilityModel(UtilityModel):
    """
    Hard Coded Utility Model class
    This model is used to compute the utility of the episodes based on hard coded values.
    It inherits from the UtilityModel class.
    """
    def __init__(self, name='utility_model', class_name='cognitive_nodes.utility_model.UtilityModel', max_iterations=20, candidate_actions=5, ltm_id="", **params):
        super().__init__(name, class_name, max_iterations, candidate_actions, ltm_id, **params)

    def setup_model(self, max_iterations, candidate_actions, ltm_id, **params):
        self.episodic_buffer = EpisodicBuffer(self, main_size=10, secondary_size=0, inputs=['perception'], outputs=[])
        self.learner = DefaultUtilityEvaluator(self, self.episodic_buffer, **params)
        self.confidence_evaluator = DefaultUtilityEvaluator(self, self.learner, self.episodic_buffer, **params)
        self.deliberation = Deliberation(self, iterations=max_iterations, candidate_actions=candidate_actions, LTM_id=ltm_id, clear_buffer=False, **params)
    
    def predict(self, input_episodes: list[Episode]) -> list[float]:
        "Work in progress"
        input_data = self.episodic_buffer.buffer_to_matrix(input_episodes, self.episodic_buffer.input_labels)
        predictions = self.learner.predict(input_data)
        if predictions is None:
            predicted_episodes = input_episodes  # If the model is not configured, return the input episodes
        else:
            self.get_logger().info(f"Predictions: {predictions}")
            self.get_logger().info(f"Output labels: {self.episodic_buffer.output_labels}")
            predicted_episodes = self.episodic_buffer.matrix_to_buffer(predictions, self.episodic_buffer.output_labels)
        self.get_logger().info(f"Prediction made: {predicted_episodes}")
        return predicted_episodes


##### LEARNERS: Place here the Learner classes that implement the learning algorithms for the Utility Model.


class DefaultUtilityModelLearner(Learner):
    """ Default Utility Model class, used when no specific utility model is defined.
    This model does not perform any learning or prediction, it simply returns a constant value.
    """
    def __init__(self, node:UtilityModel, buffer, **params):
        super().__init__(node, buffer, **params)


    def train(self):
        return None
    
    def predict(self, x):
        output_len = x.shape[0]
        y = np.ones((output_len))
        return y
    
class NoveltyUtilityModelLearner(Learner):
    """ Default Utility Model class, used when no specific utility model is defined.
    This model does not perform any learning or prediction, it simply returns a constant value.
    """
    def __init__(self, node:UtilityModel, buffer, **params):
        super().__init__(node, buffer, **params)


    def train(self):
        return None
    
    def predict(self, x):
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