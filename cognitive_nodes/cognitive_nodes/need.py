import rclpy
from rclpy.node import Node

from core.cognitive_node import CognitiveNode
from cognitive_node_interfaces.srv import SetActivation, IsSatisfied, SetWeight

import random

class Need(CognitiveNode):
    """"
    Need Class
    """
    def __init__(self, name='need', class_name = 'cognitive_nodes.need.Need', weight= 1.0, **params):
        """
        Constructor of the Need class

        Initializes a Need instance with the given name and registers it in the ltm

        :param name: The name of the Need instance
        :type name: str
        :param class_name: The name of the Need class
        :type class_name: str
        """
        super().__init__(name, class_name, **params)
        
        # N: Set Activation Service
        self.set_activation_service = self.create_service(
            SetActivation,
            'need/' + str(name) + '/set_activation',
            self.set_activation_callback
        )

        # N: Is Satisfied Service
        self.is_satisfied_service = self.create_service(
            IsSatisfied,
            'need/' + str(name) + '/get_satisfaction',
            self.get_satisfaction_callback
        )

        self.set_weight_service = self.create_service(
            SetWeight,
            'need/' + str(name) + '/set_weight',
            self.set_weight_service
        )

        self.weight = weight

    def set_weight_callback(self, request, response):
        """
        Set the weight of the need

        :param request: Request that contains the new weight of the need
        :type request: cognitive_node_interfaces.srv.SetWeight_Request
        :param response: The response indicating if the weight was set
        :type response: cognitive_node_interfaces.srv.SetWeight_Response
        :return: The response indicating if the weight was set
        :rtype: cognitive_node_interfaces.srv.SetWeight_Response
        """
        weight_value = request.weight
        self.weight = weight_value
        self.get_logger().info('Setting weight value ' + str(weight_value) + '...')
        response.set = True
        return response

    def set_activation_callback(self, request, response):
        """
        Purposes can modify the need's activation

        :param request: The request that contains the new activation value
        :type request: cognitive_node_interfaces.srv.SetActivation_Request
        :param response: The response indicating if the activation was set
        :type response: cognitive_node_interfaces.srv.SetActivation_Response
        :return: The response indicating if the activation was set
        :rtype: cognitive_node_interfaces.srv.SetActivation_Response
        """
        activation = request.activation
        self.get_logger().info('Setting activation ' + str(activation) + '...')
        self.activation = activation
        response.set = True
        return response
    
    def get_satisfaction_callback(self, request, response):
        """
        Check if the need had been satisfied

        :param request: Empty request
        :type request: cognitive_node_interfaces.srv.IsSatisfied_Request
        :param response: Response that indicates if the need is satisfied or not
        :type response: cognitive_node_interfaces.srv.IsSatisfied_Response
        :return: Response that indicates if the need is satisfied or not
        :rtype: cognitive_node_interfaces.srv.IsSatisfied_Response
        """
        self.get_logger().info('Calculating satisfaction..')
        response.satisfied = self.calculate_satisfaction()
        return response

    def calculate_satisfaction(self):
        """
        Calculate whether the need is satisfied 

        :param perception: The given normalized perception
        :type perception: dict
        :raises NotImplementedError: Evaluate method has to be implemented in a child class
        """

        raise NotImplementedError

    def calculate_activation(self, perception = None): #TODO: Implement logic
        """
        Returns the the activation value of the need

        :param perception: The given perception
        :type perception: dict
        :return: The activation of the need
        :rtype: float
        """
        self.activation = self.weight * random.random()
        if self.activation_topic:
            self.publish_activation(self.activation)
        return self.activation
    
class HeterostaticNeed(Need):
    
    def __init__(self, name='need', class_name='cognitive_nodes.need.HeterostaticNeed', weight=1.0, **params):
        super.__init__(name, class_name, weight, **params)

    def calculate_activation(self, perception=None):
        """
        Always returns an activation of 1.0

        :param perception: The given perception
        :type perception: dict
        :return: Returns the activation of the Need. Always 1.0
        :rtype: float
        """
        self.activation = 1.0 * self.weight
        if self.activation_topic:
            self.publish_activation(self.activation)
        return self.activation
    
    def calculate_satisfaction(self, perception=None):
        """
        This type of need is never satisfied, therefore always returns 0.0.

        :param perception: The given normalized perception
        :type perception: dict
        :return: False
        :rtype: bool
        """       

        return 0.0

def main(args=None):
    rclpy.init(args=args)

    need = Need()

    rclpy.spin(need)

    need.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()