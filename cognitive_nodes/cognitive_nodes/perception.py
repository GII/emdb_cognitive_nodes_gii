import rclpy
from rclpy.node import Node
from math import isclose

from core.cognitive_node import CognitiveNode
from cognitive_node_interfaces.srv import SetActivation, SetInputs
from cognitive_node_interfaces.msg import PerceptionStamped
from core.utils import class_from_classname, perception_dict_to_msg

import random

class Perception(CognitiveNode):
    """
    Perception class
    """
    def __init__(self, name='perception', class_name = 'cognitive_nodes.perception.Perception', default_msg = None, default_topic = None, normalize_data = None, **params):
        """
        Constructor for the Perception class

        Initializes a Perception instance with the given name and registers it in the ltm.
        
        :param name: The name of the Perception instance
        :type name: str
        :param class_name: The name of the Perception class
        :type class_name: str
        :param default_msg: The msg of the default subscription
        :type default_msg: str
        :param default_topic: The topic of the default subscription
        :type default_topic: str
        :param normalize_data: Values in order to normalize values
        :type normalize_data: dict
        """
        super().__init__(name, class_name, **params)       
        # We set 1.0 as the default activation value
        self.activation.activation = 1.0

        #N: Value topic
        self.perception_publisher = self.create_publisher(PerceptionStamped, "perception/" + str(name) + "/value", 0) #TODO Implement the message's publication

        # N: Set Activation Service
        self.set_activation_service = self.create_service(
            SetActivation,
            'perception/' + str(name) + '/set_activation',
            self.set_activation_callback,
            callback_group=self.cbgroup_server
        )

        # N: Set Inputs Service
        self.set_inputs_service = self.create_service(
            SetInputs,
            'perception/' + str(name) + '/set_inputs',
            self.set_inputs_callback,
            callback_group=self.cbgroup_server
        )

        self.publish_msg = PerceptionStamped()

        self.normalize_values = normalize_data

        self.default_suscription = self.create_subscription(class_from_classname(default_msg), default_topic, self.read_perception_callback, 1)
        
    def calculate_activation(self, perception = None, activation_list=None):
        """
        Returns the the activation value of the instance

        :param perception: Perception does not influence the activation of the instance
        :type perception: dict
        :return: The activation of the instance
        :rtype: float
        """
        self.activation.timestamp = self.get_clock().now().to_msg()
        return self.activation
    
    def set_activation_callback(self, request, response):
        """
        Attention mechanisms can modify the activation of a perception instance

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
    
    def set_inputs_callback(self, request, response):
        """
        Set inputs for the perception
        This method is not working yet

        :param request: The request that contains the inputs data
        :type request: cognitive_node_interfaces.SetInputs_Request
        :param response: The response that indicates if the inputs were set
        :type response: cognitive_node_interfaces.SetInputs_Response
        :return: The response that indicates if the inputs were set
        :rtype: cognitive_node_interfaces.SetInputs_Response
        """
        input_topics = request.input_topics
        input_msgs = request.input_msgs
        process_data_classes = request.process_data_classes
        self.get_logger().info('Setting inputs...' + str(input_topics) + '...')
        for input in enumerate(input_topics):
            msg_class = class_from_classname(input_msgs[input[0]])
            process_class = class_from_classname(process_data_classes[input[0]])
            self.create_subscription(msg_class, input[1], process_class.read_perception_callback())
        response.set = True
        return response
    
    def read_perception_callback(self, reading): 
        """
        Callback to process the sensor values

        :param reading: The sensor values
        :type reading: ROS msg
        """
        self.get_logger().debug("Receiving " + self.name + " = " + str(reading))
        self.reading = reading
        self.process_and_send_reading()

    def process_and_send_reading(self):
        """
        Method that processes the sensor values received
        """
        raise NotImplementedError

   
class DiscreteEventSimulatorPerception(Perception):
    """
    DiscreteEventSimulatorPerception class
    """
    def __init__(self, name='perception', class_name = 'cognitive_nodes.perception.Perception', default_msg = None, default_topic = None, normalize_data = None, **params):
        """
        Constructor for the Perception class

        Initializes a Perception instance with the given name and registers it in the ltm.
        
        :param name: The name of the Perception instance
        :type name: str
        :param class_name: The name of the Perception class
        :type class_name: str
        :param default_msg: The msg of the default subscription
        :type default_msg: str
        :param default_topic: The topic of the default subscription
        :type default_topic: str
        :param normalize_data: Values in order to normalize values
        :type normalize_data: dict
        """
        super().__init__(name, class_name, default_msg, default_topic, normalize_data, **params)
     
    def process_and_send_reading(self):
        """
        Method that processes the sensor values received
        """
        sensor = {}
        value = []
        if isinstance(self.reading.data, list):
            for perception in self.reading.data:
                distance = (
                    perception.distance - self.normalize_values["distance_min"]
                ) / (
                    self.normalize_values["distance_max"]
                    - self.normalize_values["distance_min"]
                )
                angle = (perception.angle - self.normalize_values["angle_min"]) / (
                    self.normalize_values["angle_max"]
                    - self.normalize_values["angle_min"]
                )
                diameter = (
                    perception.diameter - self.normalize_values["diameter_min"]
                ) / (
                    self.normalize_values["diameter_max"]
                    - self.normalize_values["diameter_min"]
                )
                value.append(
                    dict(
                        distance=distance,
                        angle=angle,
                        diameter=diameter,
                        # id=perception.id,
                    )
                )
        else:
            value.append(dict(data=self.reading.data))

        sensor[self.name] = value
        self.get_logger().debug("Publishing normalized " + self.name + " = " + str(sensor))
        sensor_msg = perception_dict_to_msg(sensor)
        self.publish_msg.perception=sensor_msg
        self.publish_msg.timestamp=self.get_clock().now().to_msg()
        self.perception_publisher.publish(self.publish_msg)

class Sim2DPerception(Perception):
    """
    Sim2DPerception class
    """
    def __init__(self, name='perception', class_name = 'cognitive_nodes.perception.Perception', default_msg = None, default_topic = None, normalize_data = None, **params):
        """
        Constructor for the Perception class

        Initializes a Perception instance with the given name and registers it in the ltm.
        
        :param name: The name of the Perception instance
        :type name: str
        :param class_name: The name of the Perception class
        :type class_name: str
        :param default_msg: The msg of the default subscription
        :type default_msg: str
        :param default_topic: The topic of the default subscription
        :type default_topic: str
        :param normalize_data: Values in order to normalize values
        :type normalize_data: dict
        """
        super().__init__(name, class_name, default_msg, default_topic, normalize_data, **params)
     
    def process_and_send_reading(self):
        """
        Method that processes the sensor values received
        """
        sensor = {}
        value = []
        if isinstance(self.reading.data, list):
            for perception in self.reading.data:
                x = (
                    perception.x - self.normalize_values["x_min"]
                ) / (
                    self.normalize_values["x_max"]
                    - self.normalize_values["x_min"]
                )
                y = (
                    perception.y - self.normalize_values["y_min"]
                ) / (
                    self.normalize_values["y_max"]
                    - self.normalize_values["y_min"]
                )
                if self.normalize_values.get("angle_max") and self.normalize_values.get("angle_min"):
                    angle=(
                    perception.angle - self.normalize_values["angle_min"]
                    ) / (
                        self.normalize_values["angle_max"]
                        - self.normalize_values["angle_min"]
                    )
                    value.append(
                    dict(
                        x=x,
                        y=y,
                        angle=angle
                    )
                    )
                else:
                    value.append(
                        dict(
                            x=x,
                            y=y,
                        )
                    )
        else:
            value.append(dict(data=self.reading.data))

        sensor[self.name] = value
        self.get_logger().debug("Publishing normalized " + self.name + " = " + str(sensor))
        sensor_msg = perception_dict_to_msg(sensor)
        self.publish_msg.perception=sensor_msg
        self.publish_msg.timestamp=self.get_clock().now().to_msg()
        self.perception_publisher.publish(self.publish_msg)

class IJCNNExperimentPerception(Perception):
    """IJCNN Perception class"""
    def __init__(self, name='perception', class_name = 'cognitive_nodes.perception.Perception', default_msg = None, default_topic = None, normalize_data = None, **params):
        """
        Constructor for the Perception class

        Initializes a Perception instance with the given name and registers it in the ltm.
        
        :param name: The name of the Perception instance
        :type name: str
        :param class_name: The name of the Perception class
        :type class_name: str
        :param default_msg: The msg of the default subscription
        :type default_msg: str
        :param default_topic: The topic of the default subscription
        :type default_topic: str
        :param normalize_data: Values in order to normalize values
        :type normalize_data: dict
        """
        super().__init__(name, class_name, default_msg, default_topic, normalize_data, **params)

    def process_and_send_reading(self):
        sensor = {}
        value = []
        if isinstance(self.reading.data, list):
            if "scales" in self.name:
                for perception in self.reading.data:
                    distance = (
                    perception.distance - self.normalize_values["distance_min"]
                    ) / (
                        self.normalize_values["distance_max"]
                        - self.normalize_values["distance_min"]
                    )
                    angle = (perception.angle - self.normalize_values["angle_min"]) / (
                        self.normalize_values["angle_max"]
                        - self.normalize_values["angle_min"]
                    )
                    state = perception.state/(self.normalize_values["n_states"] - 1) # Normalize 0,1,2 states between 0 and 1
                    state = 0.98 if isclose(state, 1.0) else state
                    active = perception.active
                    value.append(
                        dict(
                            distance=distance,
                            angle=angle,
                            state=state,
                            active=active
                        )
                    )
            elif "fruits" in self.name:
                for perception in self.reading.data:
                    distance = (
                    perception.distance - self.normalize_values["distance_min"]
                    ) / (
                        self.normalize_values["distance_max"]
                        - self.normalize_values["distance_min"]
                    )
                    angle = (perception.angle - self.normalize_values["angle_min"]) / (
                        self.normalize_values["angle_max"]
                        - self.normalize_values["angle_min"]
                    )
                    
                    dim_max = (
                    perception.dim_max - self.normalize_values["dim_min"]
                    ) / (
                        self.normalize_values["dim_max"]
                        - self.normalize_values["dim_min"]
                    )

                    
                    value.append(
                        dict(
                            distance = distance,
                            angle = angle,
                            dim_max = dim_max
                        )
                    )
        else:
            value.append(dict(data=self.reading.data))
        
        sensor[self.name] = value
        self.get_logger().debug("Publishing normalized " + self.name + " = " + str(sensor))
        sensor_msg = perception_dict_to_msg(sensor)
        self.publish_msg.perception=sensor_msg
        self.publish_msg.timestamp=self.get_clock().now().to_msg()
        self.perception_publisher.publish(self.publish_msg)

        



def main(args=None):
    rclpy.init(args=args)

    perception = Perception()

    rclpy.spin(perception)

    perception.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()