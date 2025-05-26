import rclpy
from copy import copy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.time import Time

from core.utils import class_from_classname
from core.cognitive_node import CognitiveNode
from core.service_client import ServiceClientAsync
from cognitive_node_interfaces.srv import SetActivation, Evaluate, GetSuccessRate, GetReward, GetSatisfaction, GetActivation
from cognitive_node_interfaces.msg import Evaluation
from builtin_interfaces.msg import Time as TimeMsg

from math import exp, isclose


class Drive(CognitiveNode):
    """
    Drive class
    """

    def __init__(self, name="drive", class_name="cognitive_nodes.drive.Drive", **params):
        """
        Constructor of the Drive class.
        Initializes a Drive instance with the given name and registers it in the LTM.

        :param name: The name of the Drive instance.
        :type name: str
        :param class_name: The name of the Drive class.
        :type class_name: str
        """
        super().__init__(name, class_name, **params)

        self.cbgroup_evaluation = MutuallyExclusiveCallbackGroup()

        # N: Set Activation Service
        self.set_activation_service = self.create_service(
            SetActivation, "drive/" + str(name) + "/set_activation", self.set_activation_callback
        )

        # N: Evaluate Service
        self.evaluation_publisher = self.create_publisher(
            Evaluation, "drive/" + str(name) + "/evaluation", 0
        )

        # N: Get Success Rate Service
        self.get_success_rate_service = self.create_service(
            GetSuccessRate,
            "drive/" + str(name) + "/get_success_rate",
            self.get_success_rate_callback,
        )

        # N: Get Reward Service
        self.get_reward_service = self.create_service(
            GetReward,
            'drive/' + str(name) + '/get_reward',
            self.get_reward_callback,
            callback_group=self.cbgroup_evaluation
        )
        

        self.evaluation_publisher_timer = self.create_timer(0.01, self.publish_evaluation_callback, callback_group = self.cbgroup_evaluation)
        self.old_evaluation=Evaluation()
        self.evaluation=Evaluation()
        self.evaluation.drive_name=self.name
        self.reward=0.0
        self.reward_timestamp=TimeMsg()
        self.configure_activation_inputs(self.neighbors)

    def set_activation_callback(self, request, response):
        """
        Needs can modify a drive's activation.

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

    def evaluate(self, perception=None):
        """
        Get expected valuation for a given perception.

        :param perception: The given normalized perception.
        :type perception: dict
        :raises NotImplementedError: Evaluate method has to be implemented in a child class.
        """
        raise NotImplementedError

    def publish_evaluation_callback(self):
        """
        Callback for publishing evaluation.
        """
        self.evaluate()
        if self.evaluation.timestamp.nanosec > 0.0:
            self.evaluation_publisher.publish(self.evaluation)

    def get_success_rate_callback(self, request, response):  # TODO: implement
        """
        Get a prediction success rate based on a historic of previous predictions.

        :param request: Empty request.
        :type request: cognitive_node_interfaces.srv.GetSuccessRate.Request
        :param response: The response that contains the predicted success rate.
        :type response: cognitive_node_interfaces.srv.GetSuccessRate.Response
        :return: The response that contains the predicted success rate.
        :rtype: cognitive_node_interfaces.srv.GetSuccessRate.Response
        """
        raise NotImplementedError
        self.get_logger().info("Getting success rate..")
        # TODO: implement logic
        response.success_rate = 0.5
        return response
    
    def get_reward_callback(self, request, response):
        """
        Callback method to calculate the reward obtained. 

        :param request: Request that includes the new perception to check the reward.
        :type request: cognitive_node_interfaces.srv.GetReward.Request
        :param response: Response that contais the reward.
        :type response: cognitive_node_interfaces.srv.GetReward.Response
        :return: Response that contais the reward.
        :rtype: cognitive_node_interfaces.srv.GetReward.Response
        """
        reward, timestamp = self.get_reward()
        response.reward = reward
        if Time.from_msg(timestamp).nanoseconds > Time.from_msg(request.timestamp).nanoseconds:
            response.updated = True
        else:
            response.updated = False
        self.get_logger().info("Obtaining reward from " + self.name + " => " + str(reward))
        return response
    
    def get_reward(self):
        """
        Returns the latest reward and its timestamp.

        :return: The latest reward and its timestamp.
        :rtype: Tuple[float, builtin_interfaces.msg.Time]
        """
        return self.reward, self.get_clock().now().to_msg()
        return self.reward, self.get_clock().now().to_msg()

    def calculate_activation(self, perception=None, activation_list=None):
        """
        Returns the the activation value of the Drive.

        :param perception: The given perception.
        :type perception: dict
        :return: The activation of the instance.
        :rtype: float
        """
        self.calculate_activation_max(activation_list)
        self.activation.activation=self.activation.activation*self.evaluation.evaluation
        timestamp_activation = Time.from_msg(self.activation.timestamp).nanoseconds
        timestamp_evaluation = Time.from_msg(self.activation.timestamp).nanoseconds
        if timestamp_evaluation<timestamp_activation:
            self.activation.timestamp = self.evaluation.timestamp
        return self.activation  
    

class DriveTopicInput(Drive):
    """
    Drive class that reads an input topic to obtain its evaluation value.
    """
    def __init__(self, name="drive", class_name="cognitive_nodes.drive.Drive", input_topic=None, input_msg=None, min_eval=0.0, **params):
        """Constructor of the DriveTopicInput class.

        Initializes a base class Drive instance and creates a subscriptor to the input topic.

        :param name: The name of the drive instance.
        :type name: str
        :param class_name: The name of the base Drive class.
        :type class_name: str
        :param input_topic: Topic where the input will be published.
        :type input_topic: str
        :param input_msg: Message type of the input topic.
        :type input_msg: ROS2 Interface
        :param min_eval: Minimum evaluation value as input reaches 1.0, defaults to 0.0.
        :type min_eval: float
        """        
        super().__init__(name, class_name, **params)
        self.min_eval=min_eval
        if input_topic:
            self.input_subscription = self.create_subscription(class_from_classname(input_msg), input_topic, self.read_input_callback, 1, callback_group=self.cbgroup_evaluation)
            self.input = 0.0
            self.input_flag = False
    
    def read_input_callback(self, msg):
        """Reads a message from the input topic and updates the evaluation and reward obtained.

        :param msg: Input data message.
        :type msg: Configurable (Typically std_msgs.msg.Float32)
        """        
        self.input = msg.data
        self.evaluate()
        self.calculate_reward()
        self.reward_timestamp=self.get_clock().now().to_msg()
        self.input_flag = True

    def calculate_reward(self):
        """
        Calculates the reward depending if the evaluation value increases or decreases.
        """        
        if self.evaluation.evaluation < self.old_evaluation.evaluation:
            self.get_logger().info(f"REWARD DETECTED. Drive: {self.name}, eval: {self.evaluation.evaluation}, old_eval: {self.old_evaluation.evaluation}")
            self.reward = 1.0
        elif self.evaluation.evaluation > self.old_evaluation.evaluation:
            self.get_logger().info(f"RESETTING REWARD. Drive: {self.name}, eval: {self.evaluation.evaluation}, old_eval: {self.old_evaluation.evaluation}")
            self.reward = 0.0

    def get_reward(self):
        """Returns the latest reward obtained.

        :return: Reward and timestamp.
        :rtype: Tuple (float, builtin_interfaces.msg.Time)
        """        
        return self.reward, self.reward_timestamp

    async def publish_activation_callback(self): #Timed publish of the activation value
        """
        Timed publish of the activation value. This method will calculate the activation based on the evaluation of the drive and the activation of its neighbors, and then publish it in the corresponding topic.
        """   
        if self.activation_topic:
            self.get_logger().debug(f'Activation Inputs: {str(self.activation_inputs)}')
            updated_activations= all((self.activation_inputs[node_name]['updated'] for node_name in self.activation_inputs))
            updated_evaluations= self.input_flag

            if updated_activations and updated_evaluations:
                self.calculate_activation(perception=None, activation_list=self.activation_inputs)
                for node_name in self.activation_inputs:
                    self.activation_inputs[node_name]['updated']=False
                self.input_flag=False
            self.publish_activation(self.activation)

    
class DriveExponential(DriveTopicInput):
    def evaluate(self, perception=None):
        """
        Evaluates the drive value according to an exponential function.

        :param perception: The given normalized perception.
        :type perception: dict
        :return: The valuation of the perception.
        :rtype: float
        """
        self.old_evaluation=copy(self.evaluation)
        if self.input>=0:
            a = 1-self.min_eval 
            self.evaluation.evaluation = a*exp(-5*self.input)+self.min_eval
            if isclose(self.input, 1.0, ):
                self.evaluation.evaluation = 0.0
        else:
            self.evaluation.evaluation = 1.0
        self.evaluation.timestamp = self.get_clock().now().to_msg()

        return self.evaluation
    

def main(args=None):
    rclpy.init(args=args)

    drive = Drive()

    rclpy.spin(drive)

    drive.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
