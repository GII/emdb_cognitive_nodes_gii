import rclpy

from core.cognitive_node import CognitiveNode
from core.utils import class_from_classname, msg_to_dict
from cognitive_node_interfaces.srv import SetActivation, Predict, GetSuccessRate, IsCompatible
from cognitive_nodes.episodic_buffer import EpisodicBuffer


class DeliberativeModel(CognitiveNode):
    """
    Deliberative Model class, this class is a generic model that can be used to implement different types of deliberative models. 
    """
    def __init__(self, name='model', class_name = 'cognitive_nodes.deliberative_model.DeliberativeModel', node_type="deliberative_model", prediction_srv_type=None, **params):
        """
        Constructor of the Deliberative Model class.

        Initializes a Deliberative instance with the given name.

        :param name: The name of the Deliberative Model instance.
        :type name: str
        :param class_name: The name of the DeliberativeModel class.
        :type class_name: str
        :param node_type: The type of the node, defaults to "deliberative_model".
        :type node_type: str
        """
        super().__init__(name, class_name, **params)

        self.episodic_buffer=None
        self.learner=None
        self.confidence_evaluator=None

        # N: Set Activation Service
        self.set_activation_service = self.create_service(
            SetActivation,
            node_type+ "/" + str(name) + '/set_activation',
            self.set_activation_callback,
            callback_group=self.cbgroup_server
        )

        # N: Predict Service
        if prediction_srv_type is not None:
            prediction_srv_type = class_from_classname(prediction_srv_type)
        else:
            raise ValueError("prediction_srv_type must be provided and be a valid class name.")
        self.predict_service = self.create_service(
            prediction_srv_type,
            node_type+ "/" + str(name) + '/predict',
            self.predict_callback,
            callback_group=self.cbgroup_server
        )

        # N: Get Success Rate Service
        self.get_success_rate_service = self.create_service(
            GetSuccessRate,
            node_type+ "/" + str(name) + '/get_success_rate',
            self.get_success_rate_callback,
            callback_group=self.cbgroup_server
        )

        # N: Is Compatible Service
        self.is_compatible_service = self.create_service(
            IsCompatible,
            node_type+ "/" + str(name) + '/is_compatible',
            self.is_compatible_callback,
            callback_group=self.cbgroup_server
        )

        #TODO: Set activation from main_loop
        #self.activation.activation = 1.0

    def set_activation_callback(self, request, response):
        """
        Some processes can modify the activation of a Model.

        :param request: The request that contains the new activation value.
        :type request: cognitive_node_interfaces.srv.SetActivation.Request
        :param response: The response indicating if the activation was set.
        :type response: cognitive_node_interfaces.srv.SetActivation.Response
        :return: The response indicating if the activation was set.
        :rtype: cognitive_node_interfaces.srv.SetActivation.Response
        """
        activation = request.activation
        self.get_logger().info('Setting activation ' + str(activation) + '...')
        self.activation.activation = activation
        self.activation.timestamp = self.get_clock().now().to_msg()
        response.set = True
        return response
    
    def predict_callback(self, request, response): # TODO: implement
        """
        Get predicted perception values for the last perceptions not newer than a given
        timestamp and for a given policy.

        :param request: The request that contains the timestamp and the policy.
        :type request: cognitive_node_interfaces.srv.Predict.Request
        :param response: The response that included the obtained perception.
        :type response: cognitive_node_interfaces.srv.Predict.Response
        :return: The response that included the obtained perception.
        :rtype: cognitive_node_interfaces.srv.Predict.Response
        """
        self.get_logger().info('Predicting ...')
        response.prediction = self.predict(request.perception, request.actuation)
        self.get_logger().info(f"Prediction made: {msg_to_dict(response.prediction)}")
        return response
    
    def get_success_rate_callback(self, request, response): # TODO: implement
        """
        Get a prediction success rate based on a historic of previous predictions.

        :param request: Empty request.
        :type request: cognitive_node_interfaces.srv.GetSuccessRate.Request
        :param response: The response that contains the predicted success rate.
        :type response: cognitive_node_interfaces.srv.GetSuccessRate.Response
        :return: The response that contains the predicted success rate.
        :rtype: cognitive_node_interfaces.srv.GetSuccessRate.Response
        """
        self.get_logger().info('Getting success rate..')
        raise NotImplementedError
        response.success_rate = 0.5
        return response
    
    def is_compatible_callback(self, request, response): # TODO: implement
        """
        Check if the Model is compatible with the current available perceptions.

        :param request: The request that contains the current available perceptions.
        :type request: cognitive_node_interfaces.srv.IsCompatible.Request
        :param response: The response indicating if the Model is compatible or not.
        :type response: cognitive_node_interfaces.srv.IsCompatible.Response
        :return: The response indicating if the Model is compatible or not.
        :rtype: cognitive_node_interfaces.srv.IsCompatible.Response
        """
        self.get_logger().info('Checking if compatible..')
        raise NotImplementedError
        response.compatible = True
        return response

    def calculate_activation(self, perception = None, activation_list=None):
        """
        Returns the activation value of the Model.

        :param perception: Perception does not influence the activation.
        :type perception: dict
        :param activation_list: List of activation values from other sources, defaults to None.
        :type activation_list: list
        :return: The activation of the instance and its timestamp.
        :rtype: cognitive_node_interfaces.msg.Activation
        """
        self.activation.timestamp = self.get_clock().now().to_msg()
        return self.activation

    def predict(self, perception, action):
        """
        Performs a prediction according to the model. See child classes.

        :param perception: Dictionary containing perception data.
        :type perception: dict
        :param action: Dictionary containing action data.
        :type action: dict
        :raises NotImplementedError: Raised when the method is not implemented.
        """
        raise NotImplementedError
    
    
    
class Learner:
    """
    Class that wraps around a learning model (Linear Classifier, ANN, SVM...)
    """    
    def __init__(self, buffer:EpisodicBuffer, **params) -> None:
        """
        Constructor of the Learner class.

        :param buffer: Episodic buffer to use.
        :type buffer: generic_model.EpisodicBuffer
        """        
        self.model=None
        self.buffer=buffer
    
    def train(self):
        """
        Placeholder method for training the model.

        :raises NotImplementedError: Not implemented yet.
        """        
        raise NotImplementedError
    
    def predict(self, perception, action):
        """
        Placeholder method for predicting an outcome.

        :param perception: Perception dictionary.
        :type perception: dict
        :param action: Candidate action dictionary.
        :type action: dict
        :raises NotImplementedError: Not implemented yet.
        """        
        raise NotImplementedError


class Evaluator:
    """
    Class that evaluates the success rate of a model based on its predictions.
    """    
    def __init__(self, node:CognitiveNode, learner:Learner,  buffer:EpisodicBuffer, **params) -> None:
        """
        Constructor of the Learner class.

        :param node: Cognitive node that uses this model.
        :type node: CognitiveNode
        :param learner: Learner instance to use for predictions.
        :type learner: Learner
        :param buffer: Episodic buffer to use.
        :type buffer: generic_model.EpisodicBuffer
        """
        self.node = node
        self.learner = learner
        self.buffer = buffer
        self.learner = learner

    def evaluate(self):
        """
        Placeholder method for evaluating the model's success rate.

        :raises NotImplementedError: Not implemented yet.
        """
        raise NotImplementedError


