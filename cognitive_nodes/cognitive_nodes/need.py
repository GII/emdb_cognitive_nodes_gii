import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.time import Time

from core.cognitive_node import CognitiveNode
from cognitive_node_interfaces.srv import SetActivation, IsSatisfied, SetWeight
from cognitive_node_interfaces.msg import Evaluation

from math import isclose

class Need(CognitiveNode):
    """"
    Need Class.
    """
    def __init__(self, name='need', class_name = 'cognitive_nodes.need.Need', weight = 1.0, drive_id = None, need_type= None, **params):
        """
        Constructor of the Need class

        Initializes a Need instance with the given name and registers it in the LTM.

        :param name: The name of the Need instance.
        :type name: str
        :param class_name: The name of the Need class.
        :type class_name: str
        :param weight: The weight of the Need.
        :type weight: float
        :param drive_id: The ID of the Drive node associated with the Need.
        :type drive_id: str
        :param need_type: The type of the Need (Operational or Cognitive).
        :type need_type: str
        """
        super().__init__(name, class_name, **params)

        self.cbgroup_satisfaction = MutuallyExclusiveCallbackGroup()
        
        # N: Set Activation Service
        self.set_activation_service = self.create_service(
            SetActivation,
            'need/' + str(name) + '/set_activation',
            self.set_activation_callback,
            callback_group= self.cbgroup_activation
        )

        # N: Is Satisfied Service
        self.is_satisfied_service = self.create_service(
            IsSatisfied,
            'need/' + str(name) + '/get_satisfaction',
            self.get_satisfaction_callback,
            callback_group=self.cbgroup_satisfaction
        )

        self.activation.activation = weight

        self.drive_id = drive_id
        self.drive_evaluation = Evaluation()
        self.drive_subscriber = self.create_subscription(Evaluation, f'drive/{self.drive_id}/evaluation', self.read_evaluation_callback, 1, callback_group=self.cbgroup_satisfaction)

        self.need_type = need_type # Need types: [Operational, Cognitive]

    def read_evaluation_callback(self, msg:Evaluation):
        """
        Callback that reads the evaluation of a Drive node. Used to check if the need is satisfied.

        :param msg: Message containing the evaluation of the Drive node.
        :type msg: cognitive_node_interfaces.msg.Evaluation
        """        
        drive_name = msg.drive_name
        if drive_name == self.drive_id:
            if Time.from_msg(msg.timestamp).nanoseconds>Time.from_msg(self.drive_evaluation.timestamp).nanoseconds:
                self.drive_evaluation = msg
            elif Time.from_msg(msg.timestamp).nanoseconds<Time.from_msg(self.drive_evaluation.timestamp).nanoseconds:
                self.get_logger().warn(f'Detected jump back in time, activation of drive evaluation: {drive_name}')
        else:
            self.get_logger().error(f'Drive evaluation mismatch detected between need {self.name} and drive (expected: {self.drive_id}, recieved {drive_name})')

    def set_activation_callback(self, request, response):
        """
        Purposes can modify the need's activation

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
    
    def get_satisfaction_callback(self, request:IsSatisfied.Request, response:IsSatisfied.Response):
        """
        Check if the need had been satisfied.

        :param request: Empty request.
        :type request: cognitive_node_interfaces.srv.IsSatisfied.Request
        :param response: Response that indicates if the need is satisfied or not.
        :type response: cognitive_node_interfaces.srv.IsSatisfied.Response
        :return: Response that indicates if the need is satisfied or not.
        :rtype: cognitive_node_interfaces.srv.IsSatisfied.Response
        """
        self.get_logger().debug('Calculating satisfaction..')
        response.satisfied = self.calculate_satisfaction()
        response.need_type = self.need_type
        if Time.from_msg(self.drive_evaluation.timestamp).nanoseconds > Time.from_msg(request.timestamp).nanoseconds:
            response.updated = True
        else:
            response.updated = False
        return response

    def calculate_satisfaction(self):
        """
        Calculate whether the need is satisfied.

        :return: True if the need is satisfied, False otherwise.
        :rtype: bool
        """
        satisfied = isclose(0, self.drive_evaluation.evaluation)

        return satisfied


    def calculate_activation(self, perception = None, activation_list=None):
        """
        Returns the the activation value of the World Model.

        :param perception: Perception does not influence the activation.
        :type perception: dict.
        :param activation_list: Activation list does not influence the activation.
        :type activation_list: list
        :return: The activation of the instance and its timestamp.
        :rtype: cognitive_node_interfaces.msg.Activation
        """
        self.activation.timestamp = self.get_clock().now().to_msg()
        return self.activation
    

def main(args=None):
    rclpy.init(args=args)

    need = Need()

    rclpy.spin(need)

    need.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()