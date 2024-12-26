import rclpy
from copy import copy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.time import Time
import threading
import inspect

from core.cognitive_node import CognitiveNode
from core.service_client import ServiceClient, ServiceClientAsync
from cognitive_node_interfaces.srv import SetActivation, IsReached, GetReward, GetActivation, Evaluate
from cognitive_node_interfaces.srv import GetIteration
from cognitive_node_interfaces.msg import Evaluation
from cognitive_processes_interfaces.msg import ControlMsg
from simulators_interfaces.srv import ObjectTooFar, CalculateClosestPosition, ObjectPickableWithTwoHands
from builtin_interfaces.msg import Time as TimeMsg

from core.utils import class_from_classname, perception_dict_to_msg, perception_msg_to_dict
from math import isclose
import numpy

import random

class Goal(CognitiveNode):
    """
    Goal class
    """
    def __init__(self, name='goal', class_name = 'cognitive_nodes.goal.Goal', **params):
        """
        Constructor of the Goal class

        Initializes a Goal with the given name and registers it in the LTM.

        :param name: The name of the Goal
        :type name: str
        :param data: More configuration info for the Goal
        :type data: dict
        :param class_name: The name of the Goal class
        :type class_name: str
        :param space_class: The class of the space used to define the Goal
        :type space_class: str
        :param space: The space used to define the Goal
        :type space: cognitive_nodes.space
        :param robot_service: The ROS service prefix to connect with robot or simulator
        :type robot_service: str
        """
        super().__init__(name, class_name, **params)
        self.reward = 0.0
        self.embedded = set()
        self.start = None
        self.end = None
        self.period = None
        self.iteration=0

        self.cbgroup_reward=MutuallyExclusiveCallbackGroup()
        
        # N: Set Activation Service
        self.set_activation_service = self.create_service(
            SetActivation,
            'goal/' + str(name) + '/set_activation',
            self.set_activation_callback, 
            callback_group=self.cbgroup_server
        )

        # N: Is Reached Service
        self.is_reached_service = self.create_service(
            IsReached,
            'goal/' + str(name) + '/is_reached',
            self.is_reached_callback,
            callback_group=self.cbgroup_server
        )

        # N: Get Reward Service
        self.get_reward_service = self.create_service(
            GetReward,
            'goal/' + str(name) + '/get_reward',
            self.get_reward_callback,
            callback_group=self.cbgroup_reward
        )

    def set_activation_callback(self, request, response):
        """
        Drives can modify a goals's activation

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
    
    async def is_reached_callback(self, request, response):
        """
        Check if the goal has been reached

        :param request: Request that includes the new perception to check
        :type request: cognitive_node_interfaces.srv.IsReached_Request
        :param response: Response that indicates if the goal is reached or not
        :type response: cognitive_node_interfaces.srv.IsReached_Response
        :return: Response that indicates if the goal is reached or not
        :rtype: cognitive_node_interfaces.srv.IsReached_Response
        """
        self.get_logger().info('Checking if is reached..')
        self.old_perception = perception_msg_to_dict(request.old_perception)
        self.perception = perception_msg_to_dict(request.perception)
        if inspect.iscoroutinefunction(self.get_reward):
            reward = await self.get_reward(self.old_perception, self.perception)
        else:
            reward = self.get_reward(self.old_perception, self.perception)
        if isclose(reward, 1.0):
            response.reached = True
        else:
            response.reached = False
        return response
    
    async def get_reward_callback(self, request, response):
        """
        Callback method to calculate the reward obtained 

        :param request: Request that includes the new perception to check the reward
        :type request: cognitive_node_interfaces.srv.GetReward_Request
        :param response: Response that contais the reward
        :type response: cognitive_node_interfaces.srv.GetReward_Response
        :return: Response that contais the reward
        :rtype: cognitive_node_interfaces.srv.GetReward_Response
        """
        self.old_perception = perception_msg_to_dict(request.old_perception)
        self.perception = perception_msg_to_dict(request.perception)
        if inspect.iscoroutinefunction(self.get_reward):
            reward, timestamp = await self.get_reward(self.old_perception, self.perception)
        else:
            reward, timestamp = self.get_reward(self.old_perception, self.perception)
        response.reward = reward
        if Time.from_msg(timestamp).nanoseconds > Time.from_msg(request.timestamp).nanoseconds:
            response.updated = True
        else:
            response.updated = False
        self.get_logger().info("Obtaining reward from " + self.name + " => " + str(reward))
        return response

    async def get_reward(self, old_perception=None, perception=None):
        """
        Calculate the reward for the current sensor values.

        This is a placeholder for the get reward method that must be implemented according
        to the required experiment/application. It is a asyncio corrutine so that service
        calls can be awaited.  

        :return: The reward obtained
        :rtype: float
        """
        raise NotImplementedError


class GoalObjectInBoxStandalone(Goal):
    """Goal representing the desire of putting an object in a box."""

    def __init__(self, name='goal', data=None, class_name='cognitive_nodes.goal.Goal', space_class=None, space=None, robot_service='simulator', normalize_data=None, **params):
        super().__init__(name, class_name, **params)
        self.robot_service = robot_service

        #Service clients
        service_name_pickable = robot_service + '/object_pickable_with_two_hands'
        self.pickable_client = ServiceClientAsync(self, ObjectPickableWithTwoHands, service_name_pickable, self.cbgroup_client)

        service_name_too_far = self.robot_service + '/object_too_far'
        self.too_far_client = ServiceClientAsync(self, ObjectTooFar, service_name_too_far, self.cbgroup_client)

        self.normalize_values=normalize_data

        if data:
            self.new_from_configuration_file(data)
        else:
            self.space = (
                space
                if space
                else class_from_classname(space_class)(ident=self.name + " space")
            )

        self.iteration_subscriber = self.create_subscription(ControlMsg, 'main_loop/control', self.get_iteration_callback, 1)

    def new_from_configuration_file(self, data):
        """
        Create attributes from the data configuration dictionary

        :param data: The configuration file
        :type data: dict
        """
        self.space = class_from_classname(data.get("space"))(ident=self.name + " space")
        self.start = data.get("start")
        self.end = data.get("end")
        self.period = data.get("period")
        for point in data.get("points", []):
            self.space.add_point(point, 1.0)

    def object_too_far(self, distance, angle):
        """
        Check is an object is too far

        :param distance: Distance of the object relative to the robot
        :type distance: float
        :param angle: Angle of the object relative to the robot
        :type angle: float
        :return: Value that indicates if the objet is too far or not
        :rtype: bool
        """
        too_far = self.too_far_client.send_request_async(distance = self.denormalize('distance', distance), angle = self.denormalize('angle', angle))
        return too_far
    
    def calculate_closest_position(self, angle):
        """
        Calculate the closest position from a given cylinder angle

        :param angle: The given angle
        :type angle: float
        :return: The closest distance and angle
        :rtype: float, float
        """
        service_name = self.robot_service + '/calculate_closest_position'
        closest_position_client = ServiceClient(CalculateClosestPosition, service_name)
        response = closest_position_client.send_request(angle = angle)
        closest_position_client.destroy_node()
        return response.dist_near, response.ang_near
    
    def object_pickable_with_two_hands_request(self, distance, angle):
        """
        Check of an obkect is pickable with the two hands of the robot

        :return: A value that indicates if the object is pickable or not
        :rtype: bool
        """
        pickable = self.pickable_client.send_request_async(distance = self.denormalize('distance', distance), angle = self.denormalize('angle', angle))
        return pickable
    
    async def object_in_close_box(self):
        """
        Check if there is an object inside of a box.

        :param perceptions: The perception given to check
        :type perceptions: dict
        :return: A value that indicates if the object is inside or not
        :rtype: bool
        """
        inside = False
        for box in self.perception["boxes"]:
            if not (await self.object_too_far(box["distance"], box["angle"])).too_far:
                for cylinder in self.perception["cylinders"]:
                    inside = (abs(box["distance"] - cylinder["distance"]) < 0.03) and (
                        abs(box["angle"] - cylinder["angle"]) < 0.02
                    )
                    if inside:
                        break
        return inside
    
    async def object_in_far_box(self):
        """
        Check if there is an object inside of a box.

        :param perceptions: The perception given to check
        :type perceptions: dict
        :return: A value that indicates if the object is inside or not
        :rtype: bool
        """
        inside = False
        for box in self.perception["boxes"]:
            if (await self.object_too_far(box["distance"], box["angle"])).too_far:
                for cylinder in self.perception["cylinders"]:
                    inside = (abs(box["distance"] - cylinder["distance"]) < 0.03) and (
                        abs(box["angle"] - cylinder["angle"]) < 0.02
                    )
                    if inside:
                        break
        return inside
    
    def object_with_robot(self):
        """
        Check if there is an object adjacent to the robot.

        :param perceptions: The perception given to check
        :type perceptions: dict
        :return: A value that indicates if the object is adjacent or not
        :rtype: bool
        """
        together = False
        if not self.object_held():
            for cylinder in self.perception["cylinders"]:
                dist_near, ang_near = self.calculate_closest_position(cylinder["angle"])
                together = (abs(cylinder["distance"] - dist_near) < 0.03) and (
                    abs(cylinder["angle"] - ang_near) < 0.02
                )
                if together:
                    break
        return together

    def object_held_with_left_hand(self):
        """
        Check if an object is held with the left hand.

        :param perceptions: The perception given to check
        :type perceptions: dict
        :return: A value that indicates if the object is held or not
        :rtype: bool
        """
        return self.perception['ball_in_left_hand'][0]['data']

    def object_held_with_right_hand(self):
        """
        Check if an object is held with the right hand.

        :param perceptions: The perception given to check
        :type perceptions: dict
        :return: A value that indicates if the object is held or not
        :rtype: bool
        """
        return self.perception['ball_in_right_hand'][0]['data']

    def object_held(self):
        """
        Check if an object is held with one hand.

        :param perceptions: The perception given to check
        :type perceptions: dict
        :return: A value that indicates if the object is held or not
        :rtype: bool
        """
        return self.object_held_with_left_hand() or self.object_held_with_right_hand()

    def object_held_before(self):
        """
        Check if an object was held with one hand.

        :param perceptions: The perception given to check
        :type perceptions: dict
        :return: A value that indicates if the object was held or not
        :rtype: bool
        """
        if self.old_perception:
            return (
                self.old_perception['ball_in_left_hand'][0]['data']
                or self.old_perception['ball_in_right_hand'][0]['data']
            )
        else:
            return False

    def object_held_with_two_hands(self):
        """
        Check if an object is held with two hands.

        :param perceptions: The perception given to check
        :type perceptions: dict
        :return: A value that indicates if the object is held or not
        :rtype: bool
        """
        return (
            self.perception['ball_in_left_hand'][0]['data']
            and self.perception['ball_in_right_hand'][0]['data']
        )

    def ball_and_box_on_the_same_side(self):
        """
        Check if an object and a box are on the same side.

        :param perceptions: The perception given to check
        :type perceptions: dict
        :return: A value that indicates if the object is in the same side or not
        :rtype: bool
        """
        same_side = False
        for box in self.perception["boxes"]:
            same_side = (self.perception['ball_in_left_hand'][0]['data'] and box['angle'] > 0.5) or (
                self.perception['ball_in_right_hand'][0]['data'] and not (box['angle'] > 0.5)
            )
            if same_side:
                break
        return same_side

    async def object_pickable_with_two_hands(self):
        """
        Check if an object can be hold with two hands.

        :param perceptions: The perception given to check
        :type perceptions: dict
        :return: A value that indicates if the object can be hold or not
        :rtype: bool
        """
        pickable = False
        for cylinder in self.perception["cylinders"]:
            pickable = (await self.object_pickable_with_two_hands_request(cylinder["distance"], cylinder["angle"])).pickable and not self.object_held()
            if pickable:
                break
        return pickable

    async def object_was_approximated(self):
        """
        Check if an object was moved towards the robot's reachable area.

        :param perceptions: The perception given to check
        :type perceptions: dict
        :return: A value that indicates if the object can be moved or not
        :rtype: bool
        """
        approximated = False
        if self.old_perception:
            for old, cur in zip(
                self.old_perception["cylinders"], self.perception["cylinders"]
            ):
                approximated = not (await self.object_too_far(
                    cur["distance"],
                    cur["angle"],
                )).too_far and (await self.object_too_far(old["distance"], old["angle"])).too_far
                if approximated:
                    break
        else:
            approximated = False
        return approximated

    def hand_was_changed(self):
        """
        Check if the held object changed from one hand to another.

        :param perceptions: The perception given to check
        :type perceptions: dict
        :return: A value that indicates if the hand changed moved or not
        :rtype: bool
        """
        return (
            (
                self.perception['ball_in_left_hand'][0]['data']
                and (not self.perception['ball_in_left_hand'][0]['data'])
            )
            and (
                (not self.old_perception['ball_in_left_hand'][0]['data'])
                and self.old_perception['ball_in_right_hand'][0]['data']
            )
        ) or (
            (
                (not self.perception['ball_in_left_hand'][0]['data'])
                and self.perception['ball_in_left_hand'][0]['data']
            )
            and (
                self.old_perception['ball_in_left_hand'][0]['data']
                and (not self.old_perception['ball_in_right_hand'][0]['data'])
            )
        )
    
    def get_iteration_callback(self, msg:ControlMsg):
        """
        Get the iteration of the experiment, if necessary

        :return: The iteration of the experiment
        :rtype: int
        """
        self.iteration=msg.iteration
        # if msg.command == "reset_world":
        #     self.perception = {}
    
    def sensorial_changes(self):
        """Return false if all perceptions have the same value as the previous step. True otherwise."""
        if not self.old_perception and self.perception:
            return True
        else:
            for sensor in self.perception:
                for perception, perception_old in zip(self.perception[sensor], self.old_perception[sensor]):
                    if isinstance(perception, dict):
                        for attribute in perception:
                            difference = abs(perception[attribute] - perception_old[attribute])
                            if difference > 0.007:
                                return True
                    else:
                        if abs(perception[0] - perception_old[0]) > 0.007:
                            return True
            return False
        
    def calculate_activation(self, perception = None, activation_list = None):
        """
        Returns the the activation value of the goal

        :param perception: Perception does not influence the activation 
        :type perception: dict
        :return: The activation of the goal
        :rtype: float
        """
        iteration=self.iteration
        if self.end:
            if(iteration % self.period >= self.start) and (
                iteration % self.period <= self.end 
            ):
                self.activation.activation = 1.0
            else:
                self.activation.activation = 0.0
        self.activation.timestamp = self.get_clock().now().to_msg()
        return self.activation

    async def get_reward(self, old_perception=None, perception=None):
        """
        Calculate the reward for the current sensor values.

        :return: The reward obtained
        :rtype: float
        """
        self.reward = 0.0
        # This is not coherent at all. I need to change it...
        # Or self.activation is not a list any longer...
        # or perceptions should be flattened
        for activation in [self.activation.activation]: #Ugly HACK: support activations as list
            if (self.sensorial_changes()) and isclose(activation, 1.0):
                if (await self.object_in_close_box()) or (await self.object_in_far_box()):
                    self.reward = 1.0
                elif self.object_held():
                    if self.object_held_with_two_hands():
                        self.reward = 0.6
                    elif self.ball_and_box_on_the_same_side():
                        self.reward = 0.6
                    elif not self.object_held_before():
                        self.reward = 0.3
                elif not self.object_held_before():
                    if (await self.object_pickable_with_two_hands()):
                        self.reward = 0.3
                    elif (await self.object_was_approximated()):
                        self.reward = 0.2
        return self.reward
    
    def denormalize(self, type, value):
        raw=0
        norm_max=0
        norm_min=0

        if not self.normalize_values:
            raise Exception('Normalization values not defined')

        if type=='distance':
            norm_max=self.normalize_values["distance_max"]
            norm_min=self.normalize_values["distance_min"]

        elif type=='angle':
            norm_max=self.normalize_values["angle_max"]
            norm_min=self.normalize_values["angle_min"]

        elif type=='diameter':
            norm_max=self.normalize_values["diameter_max"]
            norm_min=self.normalize_values["diameter_min"]

        else:
            raise ValueError
        
        raw= value*(norm_max-norm_min)+norm_min
            

        return raw
    
class GoalReadPublishedReward(Goal):
    def __init__(self, name='goal', data=None, class_name='cognitive_nodes.goal.Goal', default_topic=None, default_msg=None, **params):
        super().__init__(name, class_name, **params)
        self.reward_cbg=MutuallyExclusiveCallbackGroup()
        msg_type=class_from_classname(class_name=default_msg)
        self.reward_subscription=self.create_subscription(msg_type, default_topic, self.reward_topic_callback, 1, callback_group=self.reward_cbg)
        self.flag=threading.Event()

        self.iteration_subscriber = self.create_subscription(ControlMsg, 'main_loop/control', self.get_iteration_callback, 1)

        if data:
            self.new_from_configuration_file(data)
        else:
            self.get_logger().error(f'{self.name}: No configuration data passed to node!')


    def new_from_configuration_file(self, data):
        """
        Create attributes from the data configuration dictionary

        :param data: The configuration file
        :type data: dict
        """
        self.start = data.get("start")
        self.end = data.get("end")
        self.period = data.get("period")
        for point in data.get("points", []):
            self.space.add_point(point, 1.0)


    def reward_topic_callback(self, msg):
        self.reward=msg.data
        self.flag.set()

    def get_reward(self, old_perception=None, perception=None):
        self.flag.wait()
        reward=self.reward
        self.flag.clear()
        return reward
    
    def get_iteration_callback(self, msg:ControlMsg):
        """
        Get the iteration of the experiment, if necessary

        :return: The iteration of the experiment
        :rtype: int
        """
        self.iteration=msg.iteration
    
    def calculate_activation(self, perception = None, activation_list = None):
        """
        Returns the the activation value of the goal

        :param perception: Perception does not influence the activation 
        :type perception: dict
        :return: The activation of the goal
        :rtype: float
        """
        iteration=self.iteration
        if self.end:
            if(iteration % self.period >= self.start) and (
                iteration % self.period <= self.end 
            ):
                self.activation.activation = 1.0
            else:
                self.activation.activation = 0.0
        self.activation.timestamp = self.get_clock().now().to_msg()
        return self.activation
    
#TODO Implement GoalMotiven

class GoalMotiven(Goal):
    def __init__(self, name='goal', class_name='cognitive_nodes.goal.Goal', **params):
        super().__init__(name, class_name, **params)
        self.drive_inputs = {}
        self.old_drive_inputs = {}
        self.configure_activation_inputs(self.neighbors)
        self.configure_drive_inputs(self.neighbors)
        self.reward_timestamp=TimeMsg()
    

    def configure_drive_inputs(self, neighbors):
        drive_list = [node for node in neighbors if node['node_type']== 'Drive']
        for drive in drive_list:
            self.create_drive_input(drive)
    
    def create_drive_input(self, drive: dict):
        name = drive['name']
        node_type = drive['node_type']
        if name not in self.drive_inputs:
            if node_type == 'Drive':
                subscriber = self.create_subscription(Evaluation, 'drive/' + str(name) + '/evaluation', self.read_evaluation_callback, 1, callback_group=self.cbgroup_reward)
                data = Evaluation()
                updated = False
                new_input = dict(subscriber=subscriber, data=data, updated=updated)
                self.drive_inputs[name]=new_input
                self.get_logger().debug(f'Created new Drive input: {name}')

            else:
                self.get_logger().debug(f'Node {name} of type {node_type} is not a Drive')
        else:
            self.get_logger().error(f'Tried to add {name} to drive inputs more than once')

    def delete_drive_input(self, drive: dict):
        name = drive['name']
        if name in self.drive_inputs:
            self.destroy_subscription(self.drive_inputs[name]['subscription'])
            self.activation_inputs.pop(name)

    def add_neighbor_callback(self, request, response):
        node_name = request.neighbor_name
        node_type = request.neighbor_type
        response = super().add_neighbor_callback(request, response)
        if node_type == 'Drive':
            drive = {'name':node_name, 'node_type':node_type}
            self.create_drive_input(drive)
            response.added = True
        return response

    def delete_neighbor_callback(self, request, response):
        node_name = request.neighbor_name
        node_type = request.neighbor_type
        neighbor_to_delete = {'name':node_name, 'node_type':node_type}
        response = super().delete_neighbor_callback(request, response)

        if node_type == 'Drive':
            drive_list = [node for node in self.neighbors if node['node_type']== 'Drive']
            for drive in drive_list:
                if drive == neighbor_to_delete:
                    self.delete_drive_input(neighbor_to_delete)
                    response.deleted = True

                else:
                    response.deleted = False

        return response    

    def read_evaluation_callback(self, msg: Evaluation):
        drive_name = msg.drive_name
        if drive_name in self.drive_inputs:
            if Time.from_msg(msg.timestamp).nanoseconds>Time.from_msg(self.drive_inputs[drive_name]['data'].timestamp).nanoseconds:
                self.old_drive_inputs[drive_name] = copy(self.drive_inputs[drive_name])
                self.drive_inputs[drive_name]['data']=msg
                self.drive_inputs[drive_name]['updated']=True
                self.calculate_reward(drive_name)
                self.reward_timestamp=msg.timestamp
            elif Time.from_msg(msg.timestamp).nanoseconds<Time.from_msg(self.drive_inputs[drive_name]['data'].timestamp).nanoseconds:
                self.get_logger().warn(f'Detected jump back in time, evaluation of Drive: {drive_name}')
    
    def calculate_activation(self, perception, activation_list):
        goal_activations = {}
        goal_timestamps = {}
        for node in activation_list.keys():
            if activation_list[node]['data'].node_type == "Drive":
                goal_activations[node] = activation_list[node]['data'].activation
                goal_timestamps[node] = activation_list[node]['data'].timestamp
            if activation_list[node]['data'].node_type == "Goal":
                goal_activations[node] = activation_list[node]['data'].activation * 0.95 #Testing attenuation term, so that consequent subgoals have progressively less activation
                goal_timestamps[node] = activation_list[node]['data'].timestamp


        #THIS IS A HACK TO MAKE THE PROPAGATION OF GOAL ACTIVATION WORK. Will need to implement some sort of direction of neighbors so that downstream goals don't affect the activation of upstream goals
        if getattr(self, "act_node", None) is None:
            activation=max(zip(goal_activations.values(), goal_activations.keys()))
            timestamp=goal_timestamps[activation[1]]
            self.activation.activation = activation[0]
            self.activation.timestamp=timestamp
            self.act_node=activation[1]
        else:
            self.activation.activation = goal_activations[self.act_node]
            self.activation.timestamp=goal_timestamps[self.act_node]

    def calculate_reward(self, drive_name):
        # Remember the case in which one drive reduces its evaluation and another increases
        if self.drive_inputs[drive_name]['data'].evaluation < self.old_drive_inputs[drive_name]['data'].evaluation:
            self.get_logger().info(f"DEBUG: REWARD DETECTED. Drive: {drive_name}, eval: {self.drive_inputs[drive_name]['data'].evaluation}, old_eval: {self.old_drive_inputs[drive_name]['data'].evaluation}")
            self.reward = 1.0
        elif self.drive_inputs[drive_name]['data'].evaluation > self.old_drive_inputs[drive_name]['data'].evaluation:
            self.get_logger().info(f"DEBUG: RESETTING REWARD. Drive: {drive_name}, eval: {self.drive_inputs[drive_name]['data'].evaluation}, old_eval: {self.old_drive_inputs[drive_name]['data'].evaluation}")
            self.reward = 0.0

    def get_reward(self, old_perception=None, perception=None):
        self.get_logger().info(f"Calculating reward: {self.reward}, Drives: {self.drive_inputs}")
        reward = self.reward
        self.reward = 0.0
        return reward, self.reward_timestamp
        
def main(args=None):
    rclpy.init(args=args)

    goal = Goal()

    rclpy.spin(goal)

    goal.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()