import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.time import Time

from core.utils import class_from_classname
from core.cognitive_node import CognitiveNode
from core.service_client import ServiceClientAsync
from cognitive_node_interfaces.srv import SetActivation, Evaluate, GetSuccessRate, GetSatisfaction, GetActivation
from cognitive_node_interfaces.msg import Evaluation

from math import exp, isclose


class Drive(CognitiveNode):
    """
    Drive class
    """

    def __init__(self, name="drive", class_name="cognitive_nodes.drive.Drive", **params):
        """
        Constructor of the Drive class

        Initializes a Drive instance with the given name and registers it in the ltm


        :param name: The name of the Drive instance
        :type name: str
        :param class_name: The name of the Drive class
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

        

        self.evaluation_publisher_timer = self.create_timer(0.01, self.publish_evaluation_callback, callback_group = self.cbgroup_evaluation)

        self.evaluation=Evaluation()
        self.evaluation.drive_name=self.name
        self.configure_activation_inputs(self.neighbors)

    def set_activation_callback(self, request, response):
        """
        Needs can modify a drive's activation

        :param request: The request that contains the new activation value
        :type request: cognitive_node_interfaces.srv.SetActivation_Request
        :param response: The response indicating if the activation was set
        :type response: cognitive_node_interfaces.srv.SetActivation_Response
        :return: The response indicating if the activation was set
        :rtype: cognitive_node_interfaces.srv.SetActivation_Response
        """
        activation = request.activation
        self.get_logger().info('Setting activation ' + str(activation) + '...')
        self.activation.activation = activation
        self.activation.timestamp = self.get_clock().now().to_msg()
        response.set = True
        return response

    def evaluate(self, perception=None):
        """
        Get expected valuation for a given perception

        :param perception: The given normalized perception
        :type perception: dict
        :raises NotImplementedError: Evaluate method has to be implemented in a child class
        """
        raise NotImplementedError

    def publish_evaluation_callback(self):
        """
        Callback for publishing evaluation

        """
        self.evaluate()
        if self.evaluation.timestamp.nanosec > 0.0:
            self.evaluation_publisher.publish(self.evaluation)

    def get_success_rate_callback(self, request, response):  # TODO: implement
        """
        Get a prediction success rate based on a historic of previous predictions

        :param request: Empty request
        :type request: cognitive_node_interfaces.srv.GetSuccessRate_Request
        :param response: The response that contains the predicted success rate
        :type response: cognitive_node_interfaces.srv.GetSuccessRate_Response
        :return: The response that contains the predicted success rate
        :rtype: cognitive_node_interfaces.srv.GetSuccessRate_Response
        """
        raise NotImplementedError
        self.get_logger().info("Getting success rate..")
        # TODO: implement logic
        response.success_rate = 0.5
        return response

    def calculate_activation(self, perception=None, activation_list=None):
        """ "
        Returns the the activation value of the Drive

        :param perception: The given perception
        :type perception: dict
        :return: The activation of the instance
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
    def __init__(self, name="drive", class_name="cognitive_nodes.drive.Drive", input_topic=None, input_msg=None, min_eval=0.0, **params):
        super().__init__(name, class_name, **params)
        self.min_eval=min_eval
        if input_topic:
            self.input_subscription = self.create_subscription(class_from_classname(input_msg), input_topic, self.read_input_callback, 1)
            self.input = 0.0
            self.input_flag = False
    
    def read_input_callback(self, msg):
        self.input = msg.data
        self.evaluate()
        self.input_flag = True

    async def publish_activation_callback(self): #Timed publish of the activation value
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
        Evaluates the drive value according to the 

        :param perception: The given normalized perception
        :type perception: dict
        :return: The valuation of the perception
        :rtype: float
        """
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
