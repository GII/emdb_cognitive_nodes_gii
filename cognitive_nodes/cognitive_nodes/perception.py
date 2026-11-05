import rclpy
from rclpy.node import Node
from rclpy.logging import LoggingSeverity
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
    Base class for Perception nodes.

    A Perception node subscribes to a sensor topic, normalizes the values
    received and publishes them as a :class:`core.container.Container` message
    (:class:`core_interfaces.msg.Container`) with a single row on
    ``perception/<name>/value``. The processing is done by
    :meth:`process_and_send_reading`, which subclasses must implement. Readings are only
    processed while the activation of the node is above ``threshold``; the activation can be
    changed through the ``set_activation`` service.

    Services:

    - ``perception/<name>/set_activation``: sets the activation of the node
      (:class:`cognitive_node_interfaces.srv.SetActivation`).
    - ``perception/<name>/set_inputs``: adds subscriptions to the node
      (:class:`cognitive_node_interfaces.srv.SetInputs`). Not implemented yet.

    Topics:

    - ``perception/<name>/value``: published, :class:`core_interfaces.msg.Container`.
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
        :param default_msg: Class name of the message of the default subscription (resolved with :func:`core.utils.class_from_classname`). It is required in practice, as the subscription is always created.
        :type default_msg: str
        :param default_topic: The topic of the default subscription.
        :type default_topic: str
        :param normalize_data: Normalization limits (``<label>_min`` / ``<label>_max`` keys, depending on the subclass) used to scale the readings. It is not used by the base class.
        :type normalize_data: dict
        :param threshold: Activation threshold for processing: readings are ignored unless the activation is strictly above it. Defaults to 0.9.
        :type threshold: float
        :param params: Additional parameters passed to :class:`core.cognitive_node.CognitiveNode`.
        :type params: dict
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
        :type perception: core.container.Container or None
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
        Set inputs for the perception. Not implemented yet.

        :param request: The request that contains the inputs data.
        :type request: cognitive_node_interfaces.srv.SetInputs.Request
        :param response: The response that indicates if the inputs were set.
        :type response: cognitive_node_interfaces.srv.SetInputs.Response
        :return: The response that indicates if the inputs were set.
        :rtype: cognitive_node_interfaces.srv.SetInputs.Response
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
        Callback of the default subscription. Stores the reading and calls
        :meth:`process_and_send_reading` if the activation of the node is above ``threshold``;
        otherwise the reading is ignored.

        :param reading: The sensor values. Its ``data`` field is either a list of readings or a single value.
        :type reading: Message of the class given by ``default_msg``
        """
        if self.activation.activation > self.threshold:
            # The debug texts are built only if they are going to be logged: perceptions arrive every period
            if self.get_logger().is_enabled_for(LoggingSeverity.DEBUG):
                self.get_logger().debug("Receiving " + self.name + " = " + str(reading))
            self.reading = reading
            self.process_and_send_reading()
        else:
            self.get_logger().debug("Ignoring input...")

    def process_and_send_reading(self):
        """
        Method that processes the sensor values received in ``self.reading`` and publishes them.

        :raises NotImplementedError: This method should be implemented in subclasses.
        """
        raise NotImplementedError

   
class DiscreteEventSimulatorPerception(Perception):
    """
    Perception of the Discrete Event Simulator.

    If the reading is a list, only its first element is processed: its ``distance``,
    ``angle`` and ``diameter`` are normalized to [0, 1] using ``normalize_data``
    (keys ``distance_min/max``, ``angle_min/max``, ``diameter_min/max``) and published
    with those labels. An empty list is ignored with a warning. If the reading is not a list, its ``data`` is
    published with the label ``data``, without normalization.
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
        Normalizes the sensor values in ``self.reading`` and publishes them as a one-row
        :class:`core.container.Container` message on ``perception/<name>/value``.
        See the class documentation for the processed fields.
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

        if self.get_logger().is_enabled_for(LoggingSeverity.DEBUG):
            self.get_logger().debug("Publishing normalized " + self.name + " = " + str(self.container))
        sensor_msg = self.container.to_msg()
        self.perception_publisher.publish(sensor_msg)

class FruitShopPerception(Perception):
    """
    Perception of the Fruit Shop scenario.

    If the reading is a list, only its first element is processed. The node name selects the format:

    - Names containing ``scales``: ``distance``, ``angle``, ``state`` (scaled with ``n_states``;
      a value of 1.0 is set to 0.98) and ``active``.
    - Names containing ``fruits``: ``distance``, ``angle`` and ``dim_max`` (scaled with
      ``dim_min`` / ``dim_max``).

    Distance and angle are normalized with ``distance_min/max`` and ``angle_min/max`` from
    ``normalize_data``. If the reading is not a list, its ``data`` is published with the label ``data``.
    """
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
        Normalizes the sensor values in ``self.reading`` and publishes them as a one-row
        :class:`core.container.Container` message on ``perception/<name>/value``.
        See the class documentation for the processed fields.
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

        if self.get_logger().is_enabled_for(LoggingSeverity.DEBUG):
            self.get_logger().debug("Publishing normalized " + self.name + " = " + str(self.container))
        sensor_msg = self.container.to_msg()
        self.perception_publisher.publish(sensor_msg)



class OscarLLMPerception(Perception):
    """
    Perception of the Oscar LLM scenario.

    If the reading is a list, only its first element is processed. The node name selects the format:

    - Names containing ``object``: ``label``, ``x_position``, ``y_position`` (normalized with
      ``x_min/max``, ``y_min/max``), ``diameter`` (normalized with ``diameter_min/max``), ``color`` and ``state``.
    - Names containing ``robot_hand``: ``x_position``, ``y_position`` (normalized) and ``state``.

    If the reading is not a list, its ``data`` is published with the label ``data``.
    """
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
        Normalizes the sensor values in ``self.reading`` and publishes them as a one-row
        :class:`core.container.Container` message on ``perception/<name>/value``.
        See the class documentation for the processed fields.
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

        if self.get_logger().is_enabled_for(LoggingSeverity.DEBUG):
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