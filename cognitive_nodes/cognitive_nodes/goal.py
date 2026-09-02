import rclpy
from copy import copy
from collections import deque
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.time import Time
import numpy as np
import inspect
from math import isclose

from core.cognitive_node import CognitiveNode
from core.service_client import ServiceClient, ServiceClientAsync
from cognitive_nodes.space import PointBasedSpace
from core.utils import class_from_classname, compare_perceptions
from core.container import Container

from core_interfaces.msg import Container as ContainerMsg
from cognitive_node_interfaces.srv import AddPoints, SetActivation, IsReached, GetReward, DuplicateGoal, SendSpace, ContainsSpace
from cognitive_node_interfaces.msg import Evaluation, Perception, SuccessRate
from cognitive_processes_interfaces.msg import ControlMsg
from simulators_interfaces.srv import ObjectTooFar, CalculateClosestPosition, ObjectPickableWithTwoHands
from builtin_interfaces.msg import Time as TimeMsg


class Goal(CognitiveNode):
    """
    Goal class.
    """
    def __init__(self, name='goal', class_name = 'cognitive_nodes.goal.Goal', node_type="Goal", duplicate_from=None, **params):
        """
        Constructor of the Goal class

        Initializes a Goal with the given name and registers it in the LTM.

        :param name: The name of the Goal.
        :type name: str
        :param class_name: The name of the Goal class.
        :type class_name: str
        :param node_type: The type of the node, defaults to "Goal".
        :type node_type: str
        """
        super().__init__(name, class_name, node_type=node_type, duplicate_from=None, **params)
        self.reward = 0.0
        self.duplicate_from = duplicate_from
        if self.duplicate_from:
            service_name = f"goal/{self.duplicate_from}/duplicate_goal" 
            self.node_clients[service_name] = ServiceClientAsync(self, service_name=service_name, service_type=DuplicateGoal, callback_group=self.cbgroup_client)
        self.duplicate_count = 0
        self.base_params = params

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

        self.add_point_service = self.create_service(
            AddPoints,
            'goal/' + str(name) + '/add_points',
            self.add_points_callback,
            callback_group=self.cbgroup_server
        )

        self.send_goal_space_service = self.create_service(
            SendSpace, 
            'goal/' + str(name) + '/send_space', 
            self.send_goal_space_callback, 
            callback_group=self.cbgroup_server
        )

        self.duplicate_goal_service = self.create_service(DuplicateGoal, 'goal/' + str(
            name) + '/duplicate_goal', self.duplicate_goal_callback, callback_group=self.cbgroup_server)

    def set_activation_callback(self, request, response):
        """
        Drives can modify a goals's activation.

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
    
    def add_points_callback(self, request, response):
        """
        Callback method for adding a point (or anti-point) to a specific Goal.

        :param request: The request that contains the point that is added and its confidence.
        :type request: cognitive_node_interfaces.srv.AddPoints.Request
        :param response: The response indicating if the point was added to the Goal.
        :type response: cognitive_node_interfaces.srv.AddPoints.Response
        :return: The response indicating if the point was added to the Goal.
        :rtype: cognitive_node_interfaces.srv.AddPoints.Response
        """
        if request.points:
            points = Container.from_msg(request.points) 
            confidences = np.asarray(request.confidences)
            if len(points) != len(confidences):
                self.get_logger().error(f"Number of points and confidences do not match. Points: {len(points)}, Confidences: {len(confidences)}")
                response.added = False
                return response
            self.update_space(points, confidences)
            response.added = True
            self.get_logger().info(f'Added: {len(points)} points with mean confidence: {np.mean(confidences)}')
        else:
            response.added = False
        return response

    def send_goal_space_callback(self, request, response):
        """
        Callback that sends the goal space data.

        :param request: Empty request.
        :type request: cognitive_node_interfaces.srv.SendSpace.Request
        :param response: Response that contains the goal space data.
        :type response: cognitive_node_interfaces.srv.SendSpace.Response
        :return: Response containing the goal space data.
        :rtype: cognitive_node_interfaces.srv.SendSpace.Response
        """
        # The default goal class does not have a space, returns an empty message. See GoalLearnedSpace for an example of goal with space.
        response.space = ContainerMsg()
        return response


    async def is_reached_callback(self, request, response):
        """
        Check if the goal has been reached.

        :param request: Request that includes the new perception to check.
        :type request: cognitive_node_interfaces.srv.IsReached.Request
        :param response: Response that indicates if the goal is reached or not.
        :type response: cognitive_node_interfaces.srv.IsReached.Response
        :return: Response that indicates if the goal is reached or not.
        :rtype: cognitive_node_interfaces.srv.IsReached.Response
        """
        self.get_logger().info('Checking if is reached..')
        self.old_perception = Container.from_msg(request.old_perception)
        self.perception = Container.from_msg(request.perception)
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
        Callback method to calculate the reward obtained. 

        :param request: Request that includes the new perception to check the reward.
        :type request: cognitive_node_interfaces.srv.GetReward.Request
        :param response: Response that contais the reward.
        :type response: cognitive_node_interfaces.srv.GetReward.Response
        :return: Response that contais the reward.
        :rtype: cognitive_node_interfaces.srv.GetReward.Response
        """
        self.point_msg=request.perception
        self.old_perception = Container.from_msg(request.old_perception)
        self.perception = Container.from_msg(request.perception)
        update_space = request.update_space
        if inspect.iscoroutinefunction(self.get_reward):
            reward, timestamp = await self.get_reward(self.old_perception, self.perception, update_space=update_space)
        else:
            reward, timestamp = self.get_reward(self.old_perception, self.perception, update_space=update_space)
        response.reward = reward
        if Time.from_msg(timestamp).nanoseconds > Time.from_msg(request.timestamp).nanoseconds:
            response.updated = True
        else:
            response.updated = False
        self.get_logger().info("Obtaining reward from " + self.name + " => " + str(reward))
        return response
    
    async def duplicate_goal_callback(self, request, response):
        """
        Callback method to duplicate the goal.

        :param request: Request that includes the new perception to check the reward.
        :type request: cognitive_node_interfaces.srv.DuplicateGoal.Request
        :param response: Response that contais the name of the duplicated goal.
        :type response: cognitive_node_interfaces.srv.DuplicateGoal.Response
        :return: Response that contais the name of the duplicated goal.
        :rtype: cognitive_node_interfaces.srv.DuplicateGoal.Response
        """
        new_goal = await self.duplicate_goal()
        response.duplicate_goal_name = new_goal
        return response

    async def get_reward(self, old_perception=None, perception=None, update_space=False):
        """
        Calculate the reward for the current sensor values.

        This is a placeholder for the get reward method that must be implemented according
        to the required experiment/application. It is a asyncio corrutine so that service
        calls can be awaited.  

        :param old_perception: The previous perception data. Defaults to None.
        :type old_perception: Any
        :param perception: The current perception data. Defaults to None.
        :type perception: Any
        :raises NotImplementedError: If the method is not overridden in a subclass.
        """
        raise NotImplementedError

    
    def update_space(self, points, confidences):
        """
        Placeholder method in base goals. To be implemented in derived classes.
        
        :param points: The points that are added to the Goal.
        :type points: core.container.Container
        :param confidences: Indicates if the perception added is a point or an antipoint.
        :type confidences: float
        """
        return False
    
    async def duplicate_goal(self):
        """
        Duplicates the current goal and returns the new goal instance.

        :return: The duplicated goal instance.
        :rtype: Goal
        """
        if self.duplicate_from is None:
            new_goal = self.name + f"_dup_{self.duplicate_count}"
            self.duplicate_count += 1
            params = {"neighbors": self.neighbors, "duplicate_from": self.name, **self.base_params}
            success = await self.create_node_client(name=new_goal, class_name=self.class_name, parameters=params)
            if not success:
                self.get_logger().error(f"Failed to duplicate goal {self.name} as {new_goal}")
            self.get_logger().info(f"Duplicated goal {self.name} as {new_goal}")
        else:
            response = await self.node_clients[f"goal/{self.duplicate_from}/duplicate_goal"].send_request_async()
            new_goal = response.duplicate_goal_name
        return new_goal


class GoalObjectInBoxStandalone(Goal):
    """
    Goal representing the desire of putting an object in a box.
    
    --DEPRECATED, Use the GoalMotiven class--
    """

    def __init__(self, name='goal', data=None, class_name='cognitive_nodes.goal.Goal', space_class=None, space=None, robot_service='simulator', normalize_data=None, **params):
        """
        Constructor of the GoalObjectInBoxStandalone class.

        :param name: Name of the goal.
        :type name: str
        :param data: Configuration data for the goal.
        :type data: dict
        :param class_name: Class name of the goal, defaults to 'cognitive_nodes.goal.Goal'
        :type class_name: str
        :param space_class: Class of the space to be used.
        :type space_class: str
        :param space: Predefined space object.
        :type space: object
        :param robot_service: Name of the robot service.
        :type robot_service: str
        :param normalize_data: Normalization data for sensor values
        :type normalize_data: dict

        raises DeprecationWarning: This class is deprecated, use GoalMotiven instead.
        """
        super().__init__(name, class_name, **params)
        raise DeprecationWarning("GoalObjectInBoxStandalone is deprecated, use GoalMotiven instead")
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
                else class_from_classname(space_class)(
                    ident=self.name + " space",
                    random_seed=getattr(self, 'random_seed', 0),
                )
            )

        self.iteration_subscriber = self.create_subscription(ControlMsg, 'main_loop/control', self.get_iteration_callback, 1)

    def new_from_configuration_file(self, data):
        """
        Create attributes from the data configuration dictionary.

        :param data: The configuration file.
        :type data: dict
        """
        self.space = class_from_classname(data.get("space"))(
            ident=self.name + " space",
            random_seed=getattr(self, 'random_seed', 0),
        )
        self.start = data.get("start")
        self.end = data.get("end")
        self.period = data.get("period")
        for point in data.get("points", []):
            self.space.add_point(point, 1.0)

    def object_too_far(self, distance, angle):
        """
        Check is an object is too far.

        :param distance: Distance of the object relative to the robot.
        :type distance: float
        :param angle: Angle of the object relative to the robot.
        :type angle: float
        :return: Value that indicates if the objet is too far or not.
        :rtype: bool
        """
        too_far = self.too_far_client.send_request_async(distance = self.denormalize('distance', distance), angle = self.denormalize('angle', angle))
        return too_far
    
    def calculate_closest_position(self, angle):
        """
        Calculate the closest position from a given cylinder angle.

        :param angle: The given angle.
        :type angle: float
        :return: The closest distance and angle.
        :rtype: float, float
        """
        service_name = self.robot_service + '/calculate_closest_position'
        closest_position_client = ServiceClient(CalculateClosestPosition, service_name)
        response = closest_position_client.send_request(angle = angle)
        closest_position_client.destroy_node()
        return response.dist_near, response.ang_near
    
    def object_pickable_with_two_hands_request(self, distance, angle):
        """
        Check of an object is pickable with the two hands of the robot.

        :param angle: The distance of the object relative to the robot.
        :type angle: float
        :param angle: The angle of the object relative to the robot.
        :type angle: float
        :return: A value that indicates if the object is pickable or not.
        :rtype: bool
        """
        pickable = self.pickable_client.send_request_async(distance = self.denormalize('distance', distance), angle = self.denormalize('angle', angle))
        return pickable
    
    async def object_in_close_box(self):
        """
        Check if there is an object inside of a box.

        :return: A value that indicates if the object is inside or not.
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

        :return: A value that indicates if the object is inside or not.
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

        :return: A value that indicates if the object is adjacent or not.
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

        :return: A value that indicates if the object is held or not.
        :rtype: bool
        """
        return self.perception['ball_in_left_hand'][0]['data']

    def object_held_with_right_hand(self):
        """
        Check if an object is held with the right hand.

        :return: A value that indicates if the object is held or not.
        :rtype: bool
        """
        return self.perception['ball_in_right_hand'][0]['data']

    def object_held(self):
        """
        Check if an object is held with one hand.

        :return: A value that indicates if the object is held or not.
        :rtype: bool
        """
        return self.object_held_with_left_hand() or self.object_held_with_right_hand()

    def object_held_before(self):
        """
        Check if an object was held with one hand.

        :return: A value that indicates if the object was held or not.
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

        :return: A value that indicates if the object is held or not.
        :rtype: bool
        """
        return (
            self.perception['ball_in_left_hand'][0]['data']
            and self.perception['ball_in_right_hand'][0]['data']
        )

    def ball_and_box_on_the_same_side(self):
        """
        Check if an object and a box are on the same side.

        :return: A value that indicates if the object is in the same side or not.
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

        :return: A value that indicates if the object can be hold or not.
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

        :return: A value that indicates if the object can be moved or not.
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
        Get the iteration of the experiment, if necessary.

        :param msg: The control message containing the iteration information.
        :type msg: ControlMsg
        """
        self.iteration=msg.iteration
        # if msg.command == "reset_world":
        #     self.perception = {}
    
    def sensorial_changes(self):
        """
        Return false if all perceptions have the same value as the previous step. True otherwise.

        :return: Value that indicates if the percepction values changed or not.
        :rtype: bool
        """
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
        Returns the the activation value of the goal.

        :param perception: Perception does not influence the activation.
        :type perception: dict
        :param activation_list: List of activations. Not used.
        :type activation_list: list
        :return: The activation of the goal and its timestamp.
        :rtype: cognitive_node_interfaces.msg.Activation
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

    async def get_reward(self, old_perception=None, perception=None, update_space=False):
        """
        Calculate the reward for the current sensor values.

        :param old_perception: The previous perception. Not used.
        :type old_perception: Any
        :param perception: The current perception. Not used.
        :type perception: Any
        :return: The reward obtained.
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
        """
        Denormalize a normalized value based on the type of measurement.

        :param type: The type of measurement (e.g., 'distance', 'angle', 'diameter').
        :type type: str
        :param value: The normalized value to be denormalized.
        :type value: float
        :raises Exception: If normalization values are not defined.
        :raises ValueError: If the type is not recognized.
        :return: The denormalized value.
        :rtype: float
        """
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

class GoalMotiven(Goal):
    """
    Class that implements a Goal that aims at minimizing a drive.
    """    
    def __init__(self, name='goal', class_name='cognitive_nodes.goal.Goal', attenuation=0.7, **params):
        """
        Constructor of the GoalMotiven class.

        :param name: Name of the node.
        :type name: str
        :param class_name: The name of the base Goal class, defaults to 'cognitive_nodes.goal.Goal'.
        :type class_name: str
        :param attenuation: The attenuation factor for subgoals, defaults to 0.7.
        :type attenuation: float
        """        
        super().__init__(name, class_name, **params)
        self.attenuation = attenuation
        self.drive_inputs = {}
        self.old_drive_inputs = {}
        self.configure_activation_inputs(self.neighbors)
        self.configure_drive_inputs(self.neighbors)
        self.reward_timestamp=TimeMsg()
    

    def configure_drive_inputs(self, neighbors):
        """
        Reads the neighbors list of the goal and configures inputs for each drive.

        :param neighbors: Dictionary with the information of the node [{'name': <name>, 'node_type': <node_type>}, .... ].
        :type neighbors: dict
        """        
        drive_list = [node for node in neighbors if node['node_type']== 'Drive']
        for drive in drive_list:
            self.create_drive_input(drive)
    
    def create_drive_input(self, drive: dict):
        """
        Creates a new drive input if it does not already exist.

        :param drive: A dictionary containing the drive's name and node type.
        :type drive: dict
        :raises KeyError: If the 'name' or 'node_type' key is missing in the drive dictionary.
        """        
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
        """
        Deletes the drive input and its associated subscription.

        :param drive: The drive input to be deleted.
        :type drive: dict
        """        
        name = drive['name']
        if name in self.drive_inputs:
            self.destroy_subscription(self.drive_inputs[name]['subscription'])
            self.activation_inputs.pop(name)

    def add_neighbor_callback(self, request, response):
        """
        This method extends the base add_neighbor_callback by handling the addition of a Drive node.

        :param request: The request that contains the neighbor info.
        :type request: cognitive_node_interfaces.srv.AddNeighbor.Request
        :param response: The response that indicates if the neighbor was added.
        :type response: cognitive_node_interfaces.srv.AddNeighbor.Response
        :return: The response that indicates if the neighbor was added.
        :rtype: cognitive_node_interfaces.srv.AddNeighbor.Response
        """        
        node_name = request.neighbor_name
        node_type = request.neighbor_type
        response = super().add_neighbor_callback(request, response)
        if node_type == 'Drive':
            drive = {'name':node_name, 'node_type':node_type}
            self.create_drive_input(drive)
            response.added = True
        return response

    def delete_neighbor_callback(self, request, response):
        """
        This method extends the base delete_neighbor_callback by handling the deletion of a Drive node.

        :param request: The request that contains the neighbor info. 
        :type request: cognitive_node_interfaces.srv.DeleteNeighbor.Request
        :param response: The response that indicates if the neighbor was deleted.
        :type response: cognitive_node_interfaces.srv.DeleteNeighbor.Response
        :return: The response that indicates if the neighbor was deleted.
        :rtype: cognitive_node_interfaces.srv.DeleteNeighbor.Response
        """

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
        """
        Callback that reads the evaluation of a Drive node. It updates the reward of the goal.

        :param msg: Message containing the evaluation of the Drive node
        :type msg: cognitive_node_interfaces.msg.Evaluation
        """        
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
        """
        Calculates the activation of the goal based on the activations of the neighboring drives and goals.

        :param perception: Unused perception.
        :type perception: dict
        :param activation_list: List of activations of the neighboring nodes.
        :type activation_list: list
        """        
        goal_activations = {}
        goal_timestamps = {}
        domain_activations = {}
        for node in activation_list.keys():
            if activation_list[node]['data'].node_type == "Drive":
                goal_activations[node] = activation_list[node]['data'].activation
                goal_timestamps[node] = activation_list[node]['data'].timestamp
            if activation_list[node]['data'].node_type == "Goal":
                goal_activations[node] = activation_list[node]['data'].activation * self.attenuation
                goal_timestamps[node] = activation_list[node]['data'].timestamp
            if activation_list[node]['data'].node_type == "WorldModel":
                domain_activations[node] = activation_list[node]['data'].activation
        if goal_activations:
            activation=max(zip(goal_activations.values(), goal_activations.keys()))
            self.activation.activation=activation[0]
            self.activation.timestamp=goal_timestamps[activation[1]]
        else:
            self.activation.activation = 0.0
            self.activation.timestamp=self.get_clock().now().to_msg()

        # Check domain activation
        domain_activation_max = max(domain_activations.values()) if domain_activations else 1.0
        if isclose(domain_activation_max, 0.0):
            self.activation.activation = -1.0 # If the domain activation is zero, it means that the goal does not correspond to the current domain
            
    def calculate_reward(self, drive_name):
        """
        Calculates the reward of the goal based on the evaluation of the Drive node.

        :param drive_name: Name of the drive node.
        :type drive_name: str
        """        
        # Remember the case in which one drive reduces its evaluation and another increases
        if self.drive_inputs[drive_name]['data'].evaluation < self.old_drive_inputs[drive_name]['data'].evaluation:
            self.get_logger().info(f"REWARD DETECTED. Drive: {drive_name}, eval: {self.drive_inputs[drive_name]['data'].evaluation}, old_eval: {self.old_drive_inputs[drive_name]['data'].evaluation}")
            self.reward = 1.0
        elif self.drive_inputs[drive_name]['data'].evaluation > self.old_drive_inputs[drive_name]['data'].evaluation:
            self.get_logger().info(f"DRIVE VALUE INCREASED. Drive: {drive_name}, eval: {self.drive_inputs[drive_name]['data'].evaluation}, old_eval: {self.old_drive_inputs[drive_name]['data'].evaluation}")
            self.reward = -1.0

    def get_reward(self, old_perception=None, perception=None, update_space=False):
        """
        Returns the reward of the goal.

        :param old_perception: Unused perception, defaults to None.
        :type old_perception: Any
        :param perception: Unused perception.
        :type perception: Any.
        :return: Reward of the goal and its timestamp.
        :rtype: tuple (float, builtin_interfaces.msg.Time)
        """        
        self.get_logger().info(f"Calculating reward: {self.reward}, Drives: {self.drive_inputs}")
        reward = self.reward
        self.reward = 0.0
        return reward, self.reward_timestamp

class GoalLearnedSpace(GoalMotiven):
    """
    Class that extends the functionality of the GoalMotiven class by adding a space to store goal state space.
    """    
    def __init__(self, name='goal', class_name='cognitive_nodes.goal.Goal', space_class=None, space=None, history_size=50, min_confidence=0.85, ltm_id=None, perception=None, space_parameters=None, reward_threshold=0.8, reward_delta_threshold=0.5, low_reward_threshold=0.1, **params):
        """
        Constructor of the GoalLearnedSpace class.

        :param name: Name of the node.
        :type name: str
        :param class_name: The name of the base Goal class, defaults to 'cognitive_nodes.goal.Goal'.
        :type class_name: str
        :param space_class: Class of the space.
        :type space_class: str
        :param space: Provided space object.
        :type space: space.Space, optional
        :param history_size: Samples to consider for confidence calculation.
        :type history_size: int
        :param min_confidence: Minimum confidence to consider Goal as learned.
        :type min_confidence: float
        :param ltm_id: Id of the LTM that includes the nodes.
        :type ltm_id: str
        :param perception: Perception to add when initializing space.
        :type perception: core.container.Container
        :param space_parameters: Parameters for the space initialization.
        :type space_parameters: dict
        """        
        super().__init__(
            name,
            class_name,
            space_class=space_class,
            space=space,
            history_size=history_size,
            min_confidence=min_confidence,
            ltm_id=ltm_id,
            perception=perception,
            space_parameters=space_parameters,
            **params,
        )
        if space_class:
            # Forward the node's random_seed to the space (explicit
            # space_parameters random_seed takes precedence).
            space_kwargs = dict(space_parameters) if space_parameters else {}
            space_kwargs.setdefault('random_seed', getattr(self, 'random_seed', 0))
            self.spaces = [space if space else class_from_classname(
                    space_class)(ident=name + " space", **space_kwargs)]
            self.space=self.spaces[0]
        elif space:
            self.spaces = [space]
            self.space=space
        else:
            self.spaces = None
            self.space = None
        self.point_msg=None
        self.added_point = False
        self.LTM_id=ltm_id
        self.min_confidence=min_confidence
        self.contains_space_service = self.create_service(ContainsSpace, 'goal/' + str(
            name) + '/contains_space', self.contains_space_callback, callback_group=self.cbgroup_server)
        self.success_publisher = self.create_publisher(
            SuccessRate, f'goal/{str(name)}/confidence', 0)
        self.data_labels = []
        self.history_size = history_size
        self.history = deque([], history_size)
        self.confidence=0.0
        self.learned_space=False
        if perception:
            self._add_points(perception, 1.0)

        # Constants for reward thresholds
        self.reward_threshold = reward_threshold
        self.reward_delta_threshold = reward_delta_threshold
        self.low_reward_threshold = low_reward_threshold

    def send_goal_space_callback(self, request, response):
        """
        Callback that sends the space of the goal.

        :param request: Empty request
        :type request: cognitive_node_interfaces.srv.SendGoalSpace.Request
        :param response: Response that contains the space of the goal.
        :type response: cognitive_node_interfaces.srv.SendGoalSpace.Response
        :return: Response that contains the space of the goal.
        :rtype: cognitive_node_interfaces.srv.SendGoalSpace.Response
        """        
        if self.space:
            response.space = self.space.to_msg()

        return response

    def contains_space_callback(self, request, response):
        """
        Callback that checks if the space contains a given space.

        :param request: Request that contains the space to check.
        :type request: cognitive_node_interfaces.srv.ContainsSpace.Request
        :param response: Response that indicates if the space is contained.
        :type response: cognitive_node_interfaces.srv.ContainsSpace.Response
        :return: Response that indicates if the space is contained.
        :rtype: cognitive_node_interfaces.srv.ContainsSpace.Response
        """        
        space_data = Container.from_msg(request.space)
        if self.space and space_data is not None:
            response.contained=self.space.contains(space_data)
        else:
            response.contained=False
        return response

    def _add_points(self, point, confidences):
        """
        Add a new point (or anti-point) to the Goal.
        
        :param point: The point that is added to the Goal.
        :type point: core.container.Container
        :param confidence: Indicates if the perception added is a point or an antipoint.
        :type confidence: float
        """
        if not self.space:
            self.get_logger().error("No space defined for the Goal. Cannot add points.")
            return
        self.space.add_point(point, confidences)
        self.added_point = True

    async def get_reward(self, old_perception=None, perception=None, update_space=False):
        """
        Calculate the reward of the goal based on the perception and the reward space or the evaluation of the drive. Updates the space acording to the reward obtained.

        :param old_perception: First perception. Not used.
        :type old_perception: core.container.Container
        :param perception: Second perception. Not used.
        :type perception: core.container.Container
        :return: The reward obtained and its timestamp.
        :rtype: Tuple (float, builtin_interfaces.msg.Time)
        """
        reward=0.0
        timestamp=self.get_clock().now().to_msg()
        if not compare_perceptions(old_perception, perception):
            drive_activation, drive_timestamp = self.get_drive_activation()
            if self.linked_drive(): # If drive is linked, reward is obtained from the drive evaluation
                reward = self.reward
                self.reward = 0.0
                timestamp=self.reward_timestamp
                if update_space and not isclose(reward, 0.0): # If there is a reward or the drive is activated, we can update the space with the perception and the reward
                    self.update_space(perception, reward)
                # return reward, timestamp

            # This blocks applies in two cases:
            # 1. If the drive is not linked, we can check the expected reward from the space
            # 2. If the drive is linked but inactive, we can check the expected reward from the space.
            # In all cases, space must be learned, otherwise we cannot check the expected reward.

            if self.learned_space and isclose(drive_activation, 0.0) and isclose(reward, 0.0): # If there is no reward and the drive is not activated, we can check the expected reward from the space
                if perception:
                    expected_reward=self.get_expected_reward(perception)
                    expected_reward_old = self.get_expected_reward(old_perception)
                else:
                    return reward, timestamp
                high_exp_reward = expected_reward>self.reward_threshold
                high_reward_delta = expected_reward-expected_reward_old>self.reward_delta_threshold
                reward = 1.0 if high_exp_reward and high_reward_delta else 0.0
                timestamp = drive_timestamp #TODO Check if we should use the drive timestamp or the current time
                self.publish_success_rate()
            return reward, timestamp
        else:
            return reward, timestamp

    def get_expected_reward(self, perception:Container):
        """
        Calculate the expected reward of the goal based on the perception and the goal state.

        :param perception: One or more perceptions to evaluate.
        :type perception: core.container.Container
        :return: Expected reward for a single perception or an array of expected rewards for a batch.
        :rtype: float | numpy.ndarray
        """
        if self.space and self.added_point:
            reward_value = self.space.get_probability(perception).reshape(-1)
            reward_value = np.maximum(0.0, reward_value)
            if len(perception) > 1:
                return reward_value
            return float(reward_value[0]) if reward_value.size else 0.0
        expected_reward = 0.0 if len(perception) == 1 else np.zeros(len(perception), dtype=float)
        return expected_reward

    def get_drive_activation(self):
        """
        Get the activation of a specific drive.

        :param drive_name: Name of the drive.
        :type drive_name: str
        :return: Activation value and timestamp of the drive.
        :rtype: tuple (float, builtin_interfaces.msg.Time)
        """
        drive_name = self.get_drive()
        if not drive_name:
            self.get_logger().debug("No linked drive found for this goal.")
            return 0.0, self.get_clock().now().to_msg()
        if drive_name in self.activation_inputs:
            return self.activation_inputs[drive_name]['data'].activation, self.activation_inputs[drive_name]['data'].timestamp
        else:
            self.get_logger().error(f"Drive {drive_name} not found in activation inputs.")
            return 0.0, self.get_clock().now().to_msg()

    def update_space(self, perceptions, rewards):
        """
        Update the space of the goal based on the reward obtained and the expected reward.
        Accepts either a single perception and reward or a batch of perceptions and rewards.

        :param perceptions: Perception(s) that correspond to the reward(s) obtained.
        :type perceptions: core.container.Container
        :param rewards: Real reward(s) (from a Drive) obtained.
        :type rewards: float | list[float] | numpy.ndarray
        """
        if not self.space:
            self.get_logger().error("No space defined for the Goal. Cannot add points.")
            return

        if not isinstance(perceptions, Container):
            self.get_logger().error("Perceptions must be a Container instance.")
            return

        rewards = np.asarray(rewards, dtype=float).reshape(-1)
        if rewards.ndim != 1:
            rewards = rewards.reshape(-1)

        if perceptions.size != rewards.size:
            raise ValueError(
                f"Number of perceptions ({perceptions.size}) and rewards ({rewards.size}) do not match."
            )

        # The updates are performed according to the expected reward and the actual reward obtained. 
        # All positively rewarded perceptions are added to the space
        # Negatively rewarded perceptions are added to the space if their expected reward is above a low threshold, indicating that they were expected to be positive but were not.

        expected_rewards = self.get_expected_reward(perceptions)
        if np.isscalar(expected_rewards):
            expected_rewards = np.asarray([expected_rewards], dtype=float)
        else:
            expected_rewards = np.asarray(expected_rewards, dtype=float).reshape(-1)

        confidences = np.zeros(perceptions.size, dtype=float)
        positive_reward = rewards > 0.0
        negative_reward = rewards < -0.0
        positive_expected = expected_rewards > self.reward_threshold
        negative_not_expected = expected_rewards > self.low_reward_threshold

        confidences[positive_reward] = 1.0
        confidences[negative_reward] = -1.0

        # Perceptions with zero confidence correspond to the negatively rewarded perceptions that were expected to be negative, and are not added to the space.
        nonzero_mask = ~np.isclose(confidences, 0.0)
        if np.any(nonzero_mask):
            filtered_perceptions = Container.from_dataarray(
                perceptions.read(ordered=True).isel(sample=nonzero_mask),
                container_type=perceptions.container_type,
                name=perceptions.name,
            )
            self._add_points(filtered_perceptions, confidences[nonzero_mask])

        for idx in np.flatnonzero(positive_reward):
            self.history.appendleft(bool(positive_expected[idx])) # In positively rewarded cases, we consider the confidence to be True if the expected reward was also positive.
        for idx in np.flatnonzero(negative_not_expected):
            self.history.appendleft(False)

        self.confidence = sum(self.history) / self.history.maxlen
        # Set goal as learned if min_confidence is exceeded
        if not self.learned_space and self.confidence > self.min_confidence:
            self.learned_space = True
        # Flag goal if confidence goes below 75% of learned confidence 
        # (TODO: THIS MIGHT NOT BE THE BEST IDEA IF THERE ARE NODES ALREADY CREATED THAT DEPEND ON THE REWARDS OBTAINED FROM THE SPACE OF THIS GOAL, THEN, THE CODE IS COMMENTED OUT FOR NOW)
        # if self.learned_space and self.confidence < self.min_confidence * 0.75:
        #     self.learned_space = False
        self.get_logger().info(f"DEBUG - GOAL: {self.name} REWARD: {rewards} PRED_REWARD: {expected_rewards} CONF: {self.confidence}")
        self.publish_success_rate()

    def publish_success_rate(self):
        """
        Publish the success rate of the goal.
        """        
        msg = SuccessRate()
        msg.node_name=self.name
        msg.node_type=self.node_type
        msg.flag=self.learned_space
        msg.success_rate=self.confidence
        self.success_publisher.publish(msg)

    def linked_drive(self):
        """
        Check if there is a drived linked to the goal.

        :return: Value that indicates if there is a linked drive.
        :rtype: bool
        """        
        for neighbor in self.neighbors:
            if neighbor["node_type"]=="Drive":
                return True
        return False

    def get_drive(self):
        """
        Returns the name of the linked drive.

        :return: Drive name.
        :rtype: str
        """        
        for neighbor in self.neighbors:
            if neighbor["node_type"]=="Drive":
                return neighbor["name"]         

def main(args=None):
    rclpy.init(args=args)

    goal = Goal()

    rclpy.spin(goal)

    goal.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
