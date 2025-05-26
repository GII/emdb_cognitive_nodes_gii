import rclpy
from rclpy.node import Node
from collections import deque

from core.cognitive_node import CognitiveNode
from core.utils import class_from_classname, msg_to_dict
from cognitive_node_interfaces.srv import SetActivation, Predict, GetSuccessRate, IsCompatible


import random

class GenericModel(CognitiveNode):
    """
    Generic Model class
    """
    def __init__(self, name='model', class_name = 'cognitive_nodes.generic_model.GenericModel', node_type="generic_model", **params):
        """
        Constructor of the Generic Model class.

        Initializes a Generic instance with the given name.

        :param name: The name of the Generic Model instance.
        :type name: str
        :param class_name: The name of the GenericModel class.
        :type class_name: str
        :param node_type: The type of the node, defaults to "generic_model".
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
        self.predict_service = self.create_service(
            Predict,
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
        Check if the Model is compatible with the current avaliable perceptions.

        :param request: The request that contains the current avaliable perceptions.
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
        :return: The activation of the instance.
        :rtype: float
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
    

class EpisodicBuffer:
    """
    Class that creates a buffer of episodes to be used as a STM and learn models. 

    WORK IN PROGRESS
    """    
    def __init__(self, node:CognitiveNode, episode_topic=None, episode_msg=None, max_size=500, inputs=[], outputs=[], **params) -> None:
        """
        Constructor of the EpisodicBuffer class.

        :param node: Cognitive node that will contain this episodic buffer.
        :type node: core.cognitive_node.CognitiveNode
        :param episode_topic: Topic where episodes are read.
        :type episode_topic: str
        :param episode_msg: Message type of the episodes topic.
        :type episode_msg: str
        :param max_size: Maximum size of the episodic buffer, defaults to 500.
        :type max_size: int
        :param inputs: List to configure inputs (from the attributes of the episode message) considered in the buffer, defaults to [].
        :type inputs: list
        :param outputs: List to configure outputs (from the attributes of the episode message) considered in the buffer, defaults to [].
        :type outputs: list
        """        
        self.node=node
        self.inputs=inputs #Fields of the episode msg that are considered inputs (Used for prediction)
        self.outputs=outputs #Fields of the episode msg that are considered outputs (Predicted), or a post calculated value (e.g. Value)
        self.io_list=inputs+outputs
        self.labels=[]
        self.is_input=[]
        self.data=deque(maxlen=max_size)
        self.episode_subscription=node.create_subscription(class_from_classname(episode_msg), episode_topic, self.episode_callback, callback_group=node.cbgroup_activation)

    def configure_labels(self, msg):
        """
        Creates the label list.

        :param msg: Episode message.
        :type msg: ROS Message (most cases: cognitive_processes_interfaces.msg.Episode)
        """        
        
        for io_list, is_input_flag in [(self.inputs, True), (self.outputs, False)]:
            for io in io_list:
                io_dict = getattr(msg, io)
                for group in io_dict:
                    dims = io_dict[group][0]
                    for dim in dims:
                        self.labels.append(f"{io}:{group}:{dim}")
                        self.is_input.append(is_input_flag)

    def episode_callback(self, msg):
        """
        Callback that proccesses the episode messages.

        :param msg: Episode message.
        :type msg: ROS Message (most cases: cognitive_processes_interfaces.msg.Episode)
        """        
        if not self.labels:
            self.configure_labels(msg)
        self.process_sample(msg)

    def get_sample(self, index):
        """
        WORK IN PROGRESS: Method to obtain a sample from the buffer.

        :param index: Index of the sample to obtain.
        :type index: int
        """        
        raise NotImplementedError

    def process_sample(self, episode):
        """
        Adds a new episode to the buffer.

        :param episode: Episode message.
        :type episode: ROS Message (most cases: cognitive_processes_interfaces.msg.Episode)
        """        
        #TODO Add method so that data external to the episode can be added here. E.g. Novelty, Value...
        new=[]
        for label in self.labels:
            io, group, dim = tuple(label.split(':'))
            data_msg=getattr(episode, io) 
            data=msg_to_dict(data_msg)
            new.append(data[group][0][dim])
        self.data.appendleft(new)
            

    def split_data(self):
        """
        Work in progress.

        :raises NotImplementedError: Not implemented yet.
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
        self.training_data=[]
    
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

    

def main(args=None):
    rclpy.init(args=args)

    generic_model = GenericModel()

    rclpy.spin(generic_model)

    generic_model.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()