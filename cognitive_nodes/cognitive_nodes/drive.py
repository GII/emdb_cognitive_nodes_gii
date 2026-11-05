import rclpy
from copy import copy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.time import Time
from rclpy.logging import LoggingSeverity

from core.utils import class_from_classname, perception_msg_to_dict
from core.cognitive_node import CognitiveNode
from core.service_client import ServiceClientAsync
from cognitive_node_interfaces.srv import SetActivation, Evaluate, GetSuccessRate, GetReward, GetSatisfaction, GetActivation
from cognitive_node_interfaces.msg import Evaluation
from builtin_interfaces.msg import Time as TimeMsg

from math import exp, isclose
import re
from functools import partial

class Drive(CognitiveNode):
    """
    Base class for Drive nodes.

    A Drive evaluates how far the current situation is from the satisfaction of a motivation:
    the evaluation is 0.0 when the drive is satisfied and grows up to 1.0 when it is not. The
    evaluation is computed by :meth:`evaluate`, which subclasses must implement, and it is
    published on ``drive/<name>/evaluation`` (:class:`cognitive_node_interfaces.msg.Evaluation`)
    by a timer every 0.01 s once it has a timestamp. A Drive also keeps a reward, which the
    subclasses set when the evaluation decreases.

    Services:

    - ``drive/<name>/set_activation``: :class:`cognitive_node_interfaces.srv.SetActivation`.
    - ``drive/<name>/get_success_rate``: :class:`cognitive_node_interfaces.srv.GetSuccessRate`. Not implemented yet (raises ``NotImplementedError``).
    - ``drive/<name>/get_reward``: :class:`cognitive_node_interfaces.srv.GetReward`.

    The activation of a Drive is the maximum of the activations of its neighbors multiplied by its evaluation.
    """

    def __init__(self, name="drive", class_name="cognitive_nodes.drive.Drive", node_type="Drive", **params):
        """
        Constructor of the Drive class.
        Initializes a Drive instance with the given name and registers it in the LTM.

        :param name: The name of the Drive instance.
        :type name: str
        :param class_name: The name of the Drive class.
        :type class_name: str
        :param node_type: The type of the node, defaults to "Drive".
        :type node_type: str
        :param params: Additional parameters passed to :class:`core.cognitive_node.CognitiveNode`.
        :type params: dict
        """
        super().__init__(name, class_name, node_type=node_type, **params)

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
        Robot purposes can modify a drive's activation.

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
        Get expected valuation for a given perception. Subclasses compute here ``self.evaluation``.
        It is also called by the evaluation timer.

        :param perception: The given normalized perception.
        :type perception: core.container.Container or None
        :raises NotImplementedError: Evaluate method has to be implemented in a child class.
        """
        raise NotImplementedError

    def publish_evaluation_callback(self):
        """
        Timer callback. Calls :meth:`evaluate` and publishes ``self.evaluation`` if it already has a timestamp.
        """
        self.evaluate()
        if self.evaluation.timestamp.nanosec > 0.0:
            self.evaluation_publisher.publish(self.evaluation)

    def get_success_rate_callback(self, request, response):  # TODO: implement
        """
        Get a prediction success rate based on a historic of previous predictions.
        Not implemented yet.

        :param request: Empty request.
        :type request: cognitive_node_interfaces.srv.GetSuccessRate.Request
        :param response: The response that contains the predicted success rate.
        :type response: cognitive_node_interfaces.srv.GetSuccessRate.Response
        :return: The response that contains the predicted success rate.
        :rtype: cognitive_node_interfaces.srv.GetSuccessRate.Response
        :raises NotImplementedError: Always: the service is not implemented yet.
        """
        raise NotImplementedError
        self.get_logger().info("Getting success rate..")
        # TODO: implement logic
        response.success_rate = 0.5
        return response
    
    def get_reward_callback(self, request, response):
        """
        Callback of the ``get_reward`` service. Obtains the reward with :meth:`get_reward`.
        ``updated`` is True if the timestamp of that reward is newer than ``request.timestamp``.

        :param request: Request with the ``timestamp`` of the last reward the caller has already seen.
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
        Returns the latest reward and resets it to 0.0. The timestamp returned is the current time.

        :return: The latest reward and the current time.
        :rtype: Tuple[float, builtin_interfaces.msg.Time]
        """
        reward = self.reward
        self.reward = 0.0
        return reward, self.get_clock().now().to_msg()

    def calculate_activation(self, perception=None, activation_list=None):
        """
        Returns the the activation value of the Drive: the maximum of the activations in ``activation_list`` multiplied by the current evaluation.
        The timestamp is the older one of the activation and the evaluation.

        :param perception: The given perception. Not used.
        :type perception: core.container.Container or None
        :param activation_list: Activations of the neighbor nodes, as kept in ``activation_inputs``.
        :type activation_list: dict
        :return: The activation of the instance and its timestamp.
        :rtype: cognitive_node_interfaces.msg.Activation
        """
        self.calculate_activation_max(activation_list)
        self.activation.activation=self.activation.activation*self.evaluation.evaluation
        timestamp_activation = Time.from_msg(self.activation.timestamp).nanoseconds
        timestamp_evaluation = Time.from_msg(self.evaluation.timestamp).nanoseconds
        if timestamp_evaluation<timestamp_activation:
            self.activation.timestamp = self.evaluation.timestamp
        return self.activation
    

class DriveTopicInput(Drive):
    """
    Drive that reads a numeric input topic to obtain its evaluation. Its subclasses define
    how the input is converted into an evaluation.

    Every message received updates the evaluation and the reward. The activation is only calculated and
    published when the activations of all neighbors and a new input have arrived.
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
        :param input_msg: Class name of the message type of the input topic, resolved with :func:`core.utils.class_from_classname`. The message must have a numeric ``data`` field.
        :type input_msg: str
        :param min_eval: Minimum evaluation value as input reaches 1.0, defaults to 0.0.
        :type min_eval: float
        :param params: Additional parameters passed to :class:`Drive`.
        :type params: dict
        """        
        super().__init__(name, class_name, **params)
        self.min_eval=min_eval
        if input_topic:
            self.input_subscription = self.create_subscription(class_from_classname(input_msg), input_topic, self.read_input_callback, 1, callback_group=self.cbgroup_evaluation)
            self.input = 0.0
            self.input_flag = False
    
    def read_input_callback(self, msg):
        """Reads a message from the input topic and updates the evaluation and reward obtained.

        :param msg: Input data message, whose ``data`` field is the value of the input.
        :type msg: Configurable (typically std_msgs.msg.Float32)
        """        
        self.input = msg.data
        self.evaluate()
        self.calculate_reward()
        self.reward_timestamp=self.get_clock().now().to_msg()
        self.input_flag = True

    def calculate_reward(self):
        """
        Calculates the reward depending if the evaluation value increases or decreases:
        1.0 if the evaluation has decreased, 0.0 if it has increased, and unchanged otherwise.
        """        
        if self.evaluation.evaluation < self.old_evaluation.evaluation:
            self.get_logger().info(f"REWARD DETECTED. Drive: {self.name}, eval: {self.evaluation.evaluation}, old_eval: {self.old_evaluation.evaluation}")
            self.reward = 1.0
        elif self.evaluation.evaluation > self.old_evaluation.evaluation:
            self.get_logger().info(f"RESETTING REWARD. Drive: {self.name}, eval: {self.evaluation.evaluation}, old_eval: {self.old_evaluation.evaluation}")
            self.reward = 0.0

    def get_reward(self):
        """Returns the latest reward obtained and resets it to 0.0.

        :return: Reward and the timestamp of the last input received.
        :rtype: Tuple (float, builtin_interfaces.msg.Time)
        """
        reward = self.reward
        self.reward = 0.0
        return reward, self.reward_timestamp

    async def publish_activation_callback(self): #Timed publish of the activation value
        """
        Timed publish of the activation value. This coroutine is called by the node timer (it must be awaited if called directly). When the activations of all neighbors and a new input have been received, it calculates the activation based on the evaluation of the drive and the activation of its neighbors; the activation (updated or not) is then published in the corresponding topic.
        """   
        if self.activation_topic:
            # The text is built only if it is going to be logged: building it every period is expensive
            if self.get_logger().is_enabled_for(LoggingSeverity.DEBUG):
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
    """
    Drive whose evaluation decays exponentially with the input.

    For an input ``x >= 0`` the evaluation is ``(1 - min_eval) * exp(-5 * x) + min_eval``, and it is
    set to 0.0 when ``x`` is close to 1.0. A negative input gives an evaluation of 1.0.
    """
    def evaluate(self, perception=None):
        """
        Evaluates the drive value according to an exponential function of ``self.input``.

        :param perception: Not used: the evaluation is computed from the last input received.
        :type perception: core.container.Container or None
        :return: The evaluation and its timestamp.
        :rtype: cognitive_node_interfaces.msg.Evaluation
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

class DriveLLM(Drive):
    """
    Drive whose evaluation is given by an expression provided by an LLM (``drive_function``).

    The names of the sensors used in the expression give the perception topics
    (``/perception/<sensor>/value``) the node subscribes to. When a value has been received for
    every sensor, the expression is evaluated with ``eval`` using the attributes of the sensors
    as variables (named ``<sensor>_<attribute>``; the ``.`` of the expression is replaced by ``_``).
    """
    def __init__(self, name="drive", class_name="cognitive_nodes.drive.Drive", drive_function=None, **params):
        """Constructor of the DriveLLM class.

        Initializes a base class Drive instance and creates a subscriptor to the input topic of each sensor used in the drive function.

        :param name: The name of the drive instance.
        :type name: str
        :param class_name: The name of the base Drive class.
        :type class_name: str
        :param drive_function: Expression (provided by the LLM) that computes the drive evaluation from the sensor values.
        :type drive_function: str
        :param params: Additional parameters passed to :class:`Drive`.
        :type params: dict
        """        
        super().__init__(name, class_name, **params)
        self.drive_function = drive_function
        self.input_subscriptions = []
        self.final_values = {}

        self.updated_perceptions = {}
        self.input_flag = False
        self.topics = self.create_topics()
        if self.topics:
            for index, (topic) in enumerate(self.topics):
                topic_name = topic.split("/")[-2]
                self.updated_perceptions[topic_name] = False
                input_subscription = self.create_subscription(class_from_classname("cognitive_node_interfaces.msg.PerceptionStamped"), topic, self.read_input_callback, 1, callback_group=self.cbgroup_evaluation)
                self.input_subscriptions.append(input_subscription)

    def extract_variables(self, drive_function):
        """
        Extracts the variables (sensors and perceptions) from the drive function string.

        :param drive_function: The drive function as a string.
        :type drive_function: str
        :return: The list of sensors extracted from the drive function (every identifier in the expression, before the first ``.``, without repetitions).
        :rtype: list
        """
        pattern = r'\b[a-zA-Z_]\w*(?:\.\w+)?'
        matches = re.findall(pattern, drive_function)
        sensors = []
        perceptions = []
        for match in matches:
            perception = match
            perceptions.append(perception)
            sensor = match.split('.')[0]
            if sensor not in sensors:  
                sensors.append(sensor)
            #self.get_logger().info(f"Sensors: {sensors}, Perceptions: {perceptions}") 
        return sensors


    def create_topics(self):
        """
        Creates the list of input topics based on the drive function.
        
        :return: The list of input topics.
        :rtype: list
        """
        input_topics = []
        self.sensors = self.extract_variables(str(self.drive_function))
        for sensor in self.sensors:
            input_topic = "/perception/"+ sensor +"/value"
            input_topics.append(input_topic)
        return input_topics    
    

    async def read_input_callback(self, msg):
        """Reads a message from a perception topic. This coroutine is the subscription callback (it must be awaited if called directly).
        The values received are stored per sensor. When all the sensors have been updated, it evaluates the
        drive function, calculates the reward and marks the sensors as not updated.

        :param msg: Input data message with the perception in its ``perception`` field.
        :type msg: cognitive_node_interfaces.msg.PerceptionStamped
        """       
        perception_dict = perception_msg_to_dict(msg.perception)
        #self.get_logger().info(f"Updated perceptions: {perception_dict}")
        # Save in local dictionary {"obj1": {x:0.0, y:0.0, etc etc}} and set to true the indicator that it is updated
        for sensor in perception_dict.keys():
            for key in self.updated_perceptions.keys():
                if key == sensor:
                    self.final_values[sensor] = copy(perception_dict[sensor][0])
                    self.updated_perceptions[sensor] = True
        #self.get_logger().info(f"Updated perceptions: {self.updated_perceptions}")

        # If all the perceptions are updated, evaluate and calculate reward
        # After evaluating, set all updated_perceptions to false
        if all(self.updated_perceptions.values()):
            #self.get_logger().info(f"PERCEPTIONS!!!!!!!!!!!!!!!!: {self.updated_perceptions}")
            self.evaluate(self.final_values, self.updated_perceptions)
            self.calculate_reward()
            self.reward_timestamp=self.get_clock().now().to_msg()
            for sensor in self.updated_perceptions.keys():
                self.updated_perceptions[sensor]=False
            self.input_flag = True
    # changes format
    def extract_attributes(self):
        """
        Flatten the last values received from the sensors into a single dictionary. The ``data`` attribute of each sensor is left out.

        :return: Dictionary with an entry ``<sensor>_<attribute>`` for each attribute of each sensor in ``final_values``, which are the variables that the drive function can use.
        :rtype: dict
        """
        output={}
        for sensor in self.final_values.keys():
            for attribute in self.final_values[sensor]:
                if attribute != "data":
                    name = f"{sensor}_{attribute}"
                    output[name] = self.final_values[sensor][attribute]
        #self.get_logger().info(f"Extracted attributes: {output}")
        return output

        
    def calculate_reward(self): 
        """
        Calculates the reward depending if the evaluation value increases or decreases
        (compared with three decimals): 1.0 if it has decreased, 0.0 if it has increased, and unchanged otherwise.
        """        
        if round(self.evaluation.evaluation, 3) < round(self.old_evaluation.evaluation, 3):
            self.get_logger().info(f"REWARD DETECTED. Drive: {self.name}, eval: {self.evaluation.evaluation}, old_eval: {self.old_evaluation.evaluation}")
            self.reward = 1.0
        elif round(self.evaluation.evaluation, 3) > round(self.old_evaluation.evaluation, 3):
            self.get_logger().info(f"RESETTING REWARD. Drive: {self.name}, eval: {self.evaluation.evaluation}, old_eval: {self.old_evaluation.evaluation}")
            self.reward = 0.0

    def get_reward(self):
        """Returns the latest reward obtained, without resetting it.

        :return: Reward and the timestamp of the last evaluation.
        :rtype: Tuple (float, builtin_interfaces.msg.Time)
        """        
        return self.reward, self.reward_timestamp

    async def publish_activation_callback(self): #Timed publish of the activation value
        """
        Timed publish of the activation value. This coroutine is called by the node timer (it must be awaited if called directly). When the activations of all neighbors and a new evaluation have been received, it calculates the activation based on the evaluation of the drive and the activation of its neighbors; the activation (updated or not) is then published in the corresponding topic.
        """   
        if self.activation_topic:
            # The text is built only if it is going to be logged: building it every period is expensive
            if self.get_logger().is_enabled_for(LoggingSeverity.DEBUG):
                self.get_logger().debug(f'Activation Inputs: {str(self.activation_inputs)}')
            updated_activations= all((self.activation_inputs[node_name]['updated'] for node_name in self.activation_inputs))
            updated_evaluations= self.input_flag

            if updated_activations and updated_evaluations:
                self.calculate_activation(perception=None, activation_list=self.activation_inputs)
                for node_name in self.activation_inputs:
                    self.activation_inputs[node_name]['updated']=False
                self.input_flag=False
            self.publish_activation(self.activation)        


    def evaluate(self, final_values=None, updated_perceptions=None):
        """
        Evaluates the drive value according to the drive function provided by the LLM.
        The arguments are not used: the values come from ``self.final_values`` and ``self.updated_perceptions``.
        The evaluation is only computed if all the perceptions have been updated, and in that case
        nothing is returned (the result is stored in ``self.evaluation``).

        :param final_values: Not used.
        :type final_values: dict
        :param updated_perceptions: Not used.
        :type updated_perceptions: dict
        :return: ``self.evaluation`` if some perception has not been updated yet; None otherwise.
        :rtype: cognitive_node_interfaces.msg.Evaluation or None
        """
        if all(self.updated_perceptions.values()):
            self.old_evaluation=copy(self.evaluation)
            final_perceptions = self.extract_attributes()
            final_function = self.drive_function.replace('.', '_')
            #self.get_logger().info(f"Eval final perceptions: {final_perceptions}")
            self.evaluation.evaluation = eval(final_function, {}, final_perceptions)
            self.evaluation.timestamp = self.get_clock().now().to_msg()
        else:
            self.evaluation = self.evaluation
            return self.evaluation
            

def main(args=None):
    rclpy.init(args=args)

    drive = Drive()

    rclpy.spin(drive)

    drive.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
