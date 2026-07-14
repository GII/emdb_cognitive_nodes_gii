import rclpy
from rclpy.node import Node
import numpy as np
from math import isclose, cos, sin, pi

from core.cognitive_node import CognitiveNode
from core.container import Container
from core_interfaces.msg import Container as ContainerMsg
from cognitive_node_interfaces.srv import SetActivation, SetInputs
from core.utils import class_from_classname

import random

class Perception(CognitiveNode):
    """
    Perception class
    """
    def __init__(self, name='perception', class_name = 'cognitive_nodes.perception.Perception', node_type="Perception", default_msg = None, default_topic = None, normalize_data = None, threshold=0.9, **params):
        """
        Constructor for the Perception class.

        Initializes a Perception instance with the given name and registers it in the LTM.
        
        :param name: The name of the Perception instance.
        :type name: str
        :param class_name: The name of the Perception class.
        :type class_name: str
        :param node_type: The type of the node, defaults to "Perception".
        :type node_type: str
        :param default_msg: The msg of the default subscription.
        :type default_msg: str
        :param default_topic: The topic of the default subscription.
        :type default_topic: str
        :param normalize_data: Values in order to normalize values.
        :type normalize_data: dict
        :param threshold: The activation threshold for processing.
        :type threshold: float
        """
        super().__init__(name, class_name, node_type=node_type, **params)       
        # We set 1.0 as the default activation value
        self.activation.activation = 1.0
        #Activation threshold for processing
        self.threshold = threshold

        #N: Value topic
        self.perception_publisher = self.create_publisher(ContainerMsg, "perception/" + str(name) + "/value", 0) #TODO Implement the message's publication

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

        self.normalize_values = normalize_data

        self.default_suscription = self.create_subscription(class_from_classname(default_msg), default_topic, self.read_perception_callback, 1)
        self.container = None
        
    def calculate_activation(self, perception = None, activation_list=None):
        """
        Returns the the activation value of the instance.

        :param perception: Perception does not influence the activation of the instance.
        :type perception: dict
        :param activation_list: List of activations. Not used in this case.
        :type activation_list: list
        :return: The activation of the instance and its timestamp.
        :rtype: cognitive_node_interfaces.msg.Activation
        """
        self.activation.timestamp = self.get_clock().now().to_msg()
        return self.activation
    
    def set_activation_callback(self, request, response):
        """
        Attention mechanisms can modify the activation of a perception instance.

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
    
    def set_inputs_callback(self, request, response):
        """
        Set inputs for the perception.
        This method is not working yet.

        :param request: The request that contains the inputs data.
        :type request: cognitive_node_interfaces.SetInputs.Request
        :param response: The response that indicates if the inputs were set.
        :type response: cognitive_node_interfaces.SetInputs.Response
        :return: The response that indicates if the inputs were set.
        :rtype: cognitive_node_interfaces.SetInputs.Response
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
        Callback to process the sensor values.

        :param reading: The sensor values.
        :type reading: cognitive_node_interfaces.msg.Perception
        """
        if self.activation.activation > self.threshold:
            self.get_logger().debug("Receiving " + self.name + " = " + str(reading))
            self.reading = reading
            self.process_and_send_reading()
        else:
            self.get_logger().debug("Ignoring input...")

    def process_and_send_reading(self):
        """
        Method that processes the sensor values received.
        :raise NotImplementedError: This method should be implemented in subclasses.
        """
        raise NotImplementedError

   
class DiscreteEventSimulatorPerception(Perception):
    """
    DiscreteEventSimulatorPerception class
    """
    def __init__(self, name='perception', class_name = 'cognitive_nodes.perception.Perception', default_msg = None, default_topic = None, normalize_data = None, **params):
        """
        Constructor for the Perception class
        Initializes a Perception instance with the given name and registers it in the LTM.
        
        :param name: The name of the Perception instance.
        :type name: str
        :param class_name: The name of the Perception class.
        :type class_name: str
        :param default_msg: The msg of the default subscription.
        :type default_msg: str
        :param default_topic: The topic of the default subscription.
        :type default_topic: str
        :param normalize_data: Values in order to normalize values.
        :type normalize_data: dict
        """
        super().__init__(name, class_name, default_msg=default_msg, default_topic=default_topic, normalize_data=normalize_data, **params)
     
    def process_and_send_reading(self):
        """
        Method that processes the sensor values received.
        """
        if isinstance(self.reading.data, list):
            if len(self.reading.data) == 0:
                self.get_logger().warning("Received an empty list of perceptions.")
                return
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
                labels = ["distance", "angle", "diameter"]
                data = np.array([distance, angle, diameter])

                break # Only the first perception is processed. Legacy code used perceptions as lists with dictionaries. Now, the perception message is a labeled DataArray.
        else:
            labels = ["data"]
            data = np.array([self.reading.data])

        if self.container is None:
            self.container = Container(self.name, max_size=1, container_type="perception", labels=labels)
        self.container.push(data, labels, timestamps=self.get_clock().now().nanoseconds)

        self.get_logger().debug("Publishing normalized " + self.name + " = " + str(self.container))
        sensor_msg = self.container.to_msg()
        self.perception_publisher.publish(sensor_msg)

class FruitShopPerception(Perception):
    """Fruit Shop Perception class"""
    def __init__(self, name='perception', class_name = 'cognitive_nodes.perception.Perception', default_msg = None, default_topic = None, normalize_data = None, **params):
        """
        Constructor for the Perception class.
        Initializes a Perception instance with the given name and registers it in the LTM.
        
        :param name: The name of the Perception instance.
        :type name: str
        :param class_name: The name of the Perception class.
        :type class_name: str
        :param default_msg: The msg of the default subscription.
        :type default_msg: str
        :param default_topic: The topic of the default subscription.
        :type default_topic: str
        :param normalize_data: Values in order to normalize values.
        :type normalize_data: dict
        """
        super().__init__(name, class_name, default_msg=default_msg, default_topic=default_topic, normalize_data=normalize_data, **params)

    def process_and_send_reading(self):
        """
        Method that processes the sensor values received.
        """
        if isinstance(self.reading.data, list):
            if len(self.reading.data) == 0:
                self.get_logger().warning("Received an empty list of perceptions.")
                return
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
                    labels = ["distance", "angle", "state", "active"]
                    data = np.array([distance, angle, state, active])
                    break # Only the first perception is processed. Legacy code used perceptions as lists with dictionaries. Now, the perception message is a labeled DataArray.

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
                    labels = ["distance", "angle", "dim_max"]
                    data = np.array([distance, angle, dim_max])
                    break # Only the first perception is processed. Legacy code used perceptions as lists with dictionaries. Now, the perception message is a labeled DataArray.
        else:
            labels = ["data"]
            data = np.array([self.reading.data])
        
        if self.container is None:
            self.container = Container(self.name, max_size=1, container_type="perception", labels=labels)
        self.container.push(data, labels, timestamps=self.get_clock().now().nanoseconds)

        self.get_logger().debug("Publishing normalized " + self.name + " = " + str(self.container))
        sensor_msg = self.container.to_msg()
        self.perception_publisher.publish(sensor_msg)



class OscarLLMPerception(Perception):
    """Oscar LLM Perception class"""
    def __init__(self, name='perception', class_name = 'cognitive_nodes.perception.Perception', default_msg = None, default_topic = None, normalize_data = None, **params):
        """
        Constructor for the OscarLLMPerception class.
        Initializes a OscarLLMPerception instance with the given name and registers it in the LTM.
        
        :param name: The name of the Perception instance.
        :type name: str
        :param class_name: The name of the Perception class.
        :type class_name: str
        :param default_msg: The msg of the default subscription.
        :type default_msg: str
        :param default_topic: The topic of the default subscription.
        :type default_topic: str
        :param normalize_data: Values in order to normalize values.
        :type normalize_data: dict
        """
        super().__init__(name, class_name, default_msg=default_msg, default_topic=default_topic, normalize_data=normalize_data, **params)

    def process_and_send_reading(self):
        """
        Method that processes the sensor values received.
        """
        if isinstance(self.reading.data, list):
            if len(self.reading.data) == 0:
                self.get_logger().warning("Received an empty list of perceptions.")
                return
            if "object" in self.name:
                for perception in self.reading.data:
                    label = perception.label
                    x_position = (perception.x_position
                                  - self.normalize_values["x_min"]
                                  ) / (
                        self.normalize_values["x_max"]
                        - self.normalize_values["x_min"]
                    )
                    y_position = (perception.y_position 
                                  - self.normalize_values["y_min"]) / (
                        self.normalize_values["y_max"]
                        - self.normalize_values["y_min"]
                    )
                    diameter = (perception.diameter - self.normalize_values["diameter_min"]) / (
                        self.normalize_values["diameter_max"]
                        - self.normalize_values["diameter_min"]
                    )
                    color = (perception.color
                    )
                    state = perception.state
                    labels = ["label", "x_position", "y_position", "diameter", "color", "state"]
                    data = np.array([label, x_position, y_position, diameter, color, state])
                    break # Only the first perception is processed. Legacy code used perceptions as lists with dictionaries. Now, the perception message is a labeled DataArray.
            
            elif "robot_hand" in self.name:
                for perception in self.reading.data:
                    state = perception.state
                    x = (perception.x_position 
                         - self.normalize_values["x_min"]                     
                    ) / (
                        self.normalize_values["x_max"] 
                        - self.normalize_values["x_min"]
                    )
                    y = (perception.y_position
                         - self.normalize_values["y_min"]                   
                    ) / (
                        self.normalize_values["y_max"]
                        - self.normalize_values["y_min"]
                        )
                    labels = ["x_position", "y_position", "state"]
                    data = np.array([x, y, state])
                    break # Only the first perception is processed. Legacy code used perceptions as lists with dictionaries. Now, the perception message is a labeled DataArray.
        else:
            labels = ["data"]
            data = np.array([self.reading.data])
        
        if self.container is None:
            self.container = Container(self.name, max_size=1, container_type="perception", labels=labels)
        self.container.push(data, labels, timestamps=self.get_clock().now().nanoseconds)

        self.get_logger().debug("Publishing normalized " + self.name + " = " + str(self.container))
        sensor_msg = self.container.to_msg()
        self.perception_publisher.publish(sensor_msg)

def main(args=None):
    rclpy.init(args=args)

    perception = Perception()

    rclpy.spin(perception)

    perception.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()