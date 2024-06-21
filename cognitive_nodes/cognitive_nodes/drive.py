import rclpy
from rclpy.node import Node

from core.cognitive_node import CognitiveNode
from core.service_client import ServiceClientAsync
from cognitive_node_interfaces.srv import SetActivation, Evaluate, GetSuccessRate, GetSatisfaction, GetActivation

import random


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

        # N: Set Activation Service
        self.set_activation_service = self.create_service(
            SetActivation, "drive/" + str(name) + "/set_activation", self.set_activation_callback
        )

        # N: Evaluate Service
        self.evaluate_service = self.create_service(
            Evaluate, "drive/" + str(name) + "/evaluate", self.evaluate_callback
        )

        # N: Get Success Rate Service
        self.get_success_rate_service = self.create_service(
            GetSuccessRate,
            "drive/" + str(name) + "/get_success_rate",
            self.get_success_rate_callback,
        )

        self.evaluation=0.0

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
        self.get_logger().info("Setting activation " + str(activation) + "...")
        self.activation = activation
        response.set = True
        return response

    def evaluate(self, perception):
        """
        Get expected valuation for a given perception

        :param perception: The given normalized perception
        :type perception: dict
        :raises NotImplementedError: Evaluate method has to be implemented in a child class
        """
        raise NotImplementedError

    def evaluate_callback(self, request, response):
        """
        Callback for evaluate a perception

        :param request: The request that contains the perception
        :type request: cognitive_node_interfaces.srv.Evaluate_Request
        :param response: The response that contains tha valuation of the perception
        :type response: cognitive_node_interfaces.srv.Evaluate_Response
        :return: The response that contains the valuation of the perception
        :rtype: cognitive_node_interfaces.srv.Evaluate_Response
        """
        perception, weight = request.perception, request.weight
        self.get_logger().info("Evaluating for perception " + str(perception) + "...")
        response.valuation = self.evaluate(perception, weight)
        return response

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
        self.get_logger().info("Getting success rate..")
        # TODO: implement logic
        response.success_rate = 0.5
        return response

    def calculate_activation(self, perception=None):  # TODO: Implement logic
        """ "
        Returns the the activation value of the Drive

        :param perception: The given perception
        :type perception: dict
        :return: The activation of the instance
        :rtype: float
        """
        self.activation = random.random()
        if self.activation_topic:
            self.publish_activation(self.activation)
        return self.activation  

class NeedLinkedDrive(Drive):
    """
    Need Linked Drive class: Defines a drive that is connected to one and only one need.
    """

    def __init__(self, name="drive", class_name="cognitive_nodes.drive.DriveLinkedNeed", **params):
        super().__init__(name, class_name, **params)

        need= [neighbor["name"] for neighbor in self.neighbors if neighbor["node_type"]=="Need"]
        if len(need)!=1:
            raise Exception(f'Drive must be linked to one and only one need. Connected needs: {len(need)}')
        else:
            self.cli_need_activation=ServiceClientAsync(self, GetActivation, f'cognitive_node/{need[0]}/get_activation', self.cbgroup_client)
            self.cli_need_satisfied=ServiceClientAsync(self, GetSatisfaction, f'need/{need[0]}/get_satisfaction', self.cbgroup_client)
        
    def calculate_activation(self, perception=None):
        """
        Calculates the activation of the drive. Must be implemented in a child class

        :param perception: The given normalized perception
        :type perception: dict
        :return: The valuation of the perception
        :rtype: float
        """
        raise NotImplementedError
    
class DriveLinear(NeedLinkedDrive):
    def __init__(self, name="drive", class_name="cognitive_nodes.drive.DriveLinear", **params):
        super().__init__(name, class_name, **params)  
    
    async def calculate_activation(self, perception=None):
        """
        Cascades the activation from the corresponding need

        :param perception: The given normalized perception
        :type perception: dict
        :return: The valuation of the perception
        :rtype: float
        """
        need_activation= await self.cli_need_activation.send_request_async(perception)
        assert isinstance(need_activation, GetActivation.Response)
        self.activation=need_activation
        if self.activation_topic:
            self.publish_activation(self.activation)
        return self.activation
    

    async def evaluate(self, perception=None):
        """
        Evaluates the drive value according to the 

        :param perception: The given normalized perception
        :type perception: dict
        :return: The valuation of the perception
        :rtype: float
        """
        need_satisfaction= await self.cli_need_satisfied.send_request_async(perception)
        assert isinstance(need_satisfaction, GetSatisfaction.Response)
        self.evaluation=need_satisfaction.weight*(1-need_satisfaction.satisfied)
        return self.evaluation
    

def main(args=None):
    rclpy.init(args=args)

    drive = Drive()

    rclpy.spin(drive)

    drive.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
