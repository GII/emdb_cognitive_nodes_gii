import yaml
import numpy as np

from math import isclose
from copy import copy

from cognitive_nodes.drive import Drive
from cognitive_nodes.goal import GoalLearnedSpace
from cognitive_nodes.policy import Policy
from core.service_client import ServiceClientAsync
from cognitive_nodes.episode import container_msg_to_episode, episode_obj_to_msg
from core.utils import compare_perceptions

from cognitive_node_interfaces.msg import SuccessRate
from cognitive_node_interfaces.srv import GetActivation, SendSpace, GetEffects, ContainsSpace, AddPoints, LogExecution
from cognitive_nodes.utils import PNodeSuccess, EpisodeSubscription



class DriveEffectanceInternal(Drive, PNodeSuccess):
    """Drive that detects internal architecture effects. In this case, the consolidation of any P-Node in the architecture. 

    This class inherits from the general Drive class and PNodeSuccess class, which provides helper methods to subscribe to P-Nodes success rate.
    """    
    def __init__(self, name="drive_effectance", class_name="cognitive_nodes.drive.Drive", ltm_id=None, min_confidence=0.1, limit_depth=False, **params):
        """Constructor of the DriveEffectanceInternal class.

        :param name: Name of the node.
        :type name: str
        :param class_name: The name of the base Drive class, defaults to "cognitive_nodes.drive.Drive".
        :type class_name: str
        :param ltm_id: Id of the LTM that includes the nodes.
        :type ltm_id: str
        :param min_confidence: Confidence level where P-Nodes are considered learned.
        :type min_confidence: float
        :param limit_depth: If true, excludes from analysis P-Nodes that have resulted from a sub-goal related to another P-Node.
        :type limit_depth: bool
        :raises Exception: Raises an exception if no LTM name is provided.
        """        
        super().__init__(name, class_name, **params)
        if ltm_id is None:
            raise Exception('No LTM input was provided.')
        else:    
            self.LTM_id = ltm_id
        self.limit_depth=limit_depth
        if self.limit_depth:
            self.get_logger().error("DEBUG MESSAGE - DEPTH LIMIT ACTIVE")
        self.min_confidence=min_confidence
        self.configure_pnode_success(self.LTM_id, self.cbgroup_client)

    
    def pnode_success_callback(self, msg):
        """
        Callback that proccesses a success message from a P-Node.

        :param msg: Message with success information.
        :type msg: cognitive_node_interfaces.msg.SuccessRate
        """        

        #UGLY HACK: This was done to limit effectance chains to a depth of 1. 
        # This must be done properly by analyzing neighbor chains and be general for any desired depth
        if not self.limit_depth:
            return super().pnode_success_callback(msg)
        else:
            pnode = msg.node_name
            goal_linked = msg.flag
            success_rate = msg.success_rate
            if "reach_pnode_" in pnode:
                self.pnode_evaluation[pnode] = 0.0
            else:
                self.pnode_evaluation[pnode] = success_rate * (not goal_linked)


    def evaluate(self):
        """
        Calculates drive evaluation. If any P-Node is above the minimum confidence, drive evaluation is 1.0.

        :return: Drive evaluation and its timestamp.
        :rtype: cognitive_node_interfaces.Evaluation
        """        
        max_pnode= max(self.pnode_evaluation.values(), default=0.0)
        if max_pnode>=self.min_confidence:
            self.evaluation.evaluation = 1.0
        else:
            self.evaluation.evaluation = 0.0
        self.evaluation.timestamp = self.get_clock().now().to_msg()
        return self.evaluation
    
class DriveEffectanceExternal(Drive, EpisodeSubscription):
    """
    Drive that detects effects in the environment. In this case, changes from 0 to 1 in a sensor. 

    This class inherits from the general Drive class and EpisodeSubscription class, which provides helper methods to subscribe to the episodes topic of a cognitive process.
    """
    def __init__(self, name="drive_effectance", class_name="cognitive_nodes.drive.Drive", episodes_topic=None, episodes_msg=None, **params):
        """
        Constructor of the DriveEffectanceExternal class.

        :param name: Name of the node.
        :type name: str
        :param class_name: The name of the base Drive class, defaults to "cognitive_nodes.drive.Drive"
        :type class_name: str
        :param episodes_topic: Topic from where to read the episodes.
        :type episodes_topic: str
        :param episodes_msg: Message type of the episodes topic (most cases: cognitive_node_interfaces.msg.Episode).
        :type episodes_msg: str
        :raises Exception: Raises exception if no episode topic was provided.
        """        
        super().__init__(name, class_name, **params)
        if not episodes_topic or not episodes_msg:
            raise Exception('No episode input was provided.')
        else:    
            self.episode_topic = episodes_topic
            self.episode_msg = episodes_msg
        self.effects={} 
        self.new_effects={}
        self.new_effects_data={}
        self.get_effects_service = self.create_service(GetEffects, 'drive/' + str(
            name) + '/get_effects', self.get_effects_callback, callback_group=self.cbgroup_server)
        self.configure_episode_subscription(episodes_topic, episodes_msg, self.cbgroup_activation)
    
    def episode_callback(self, msg):
        """
        Callback that processes an episode message.

        :param msg: Episode message.
        :type msg: ROS Message (most cases: cognitive_node_interfaces.msg.Episode)
        """
        episode = container_msg_to_episode(msg)
        self.find_effects(episode)

    def find_effects(self, episode):
        """
        Checks consecutive perceptions if effects were generated.

        :param perception: Current perception.
        :type perception: core.container.Container
        :param old_perception: Previous perception.
        :type old_perception: core.container.Container
        """
        old_perception = episode.old_perception
        perception = episode.perception
        labels_old_perception = set(old_perception.feature_labels)
        labels_perception = set(perception.feature_labels)
        labels = list(labels_perception.intersection(labels_old_perception))

        data_old_perception = old_perception.read().sel(features=labels).values
        data_perception = perception.read().sel(features=labels).values

        # Work with the last sample (normally only one sample is present).
        data_old = np.asarray(data_old_perception)
        data_new = np.asarray(data_perception)

        if data_old.size == 0 or data_new.size == 0:
            return

        # Take last row (safe for 1D or 2D arrays)
        row_old = data_old[-1] if data_old.ndim > 1 else data_old
        row_new = data_new[-1] if data_new.ndim > 1 else data_new

        row_old = np.asarray(row_old).reshape(-1)
        row_new = np.asarray(row_new).reshape(-1)

        # Element-wise difference and find exact 0->1 transitions
        diff = row_new - row_old
        matches = np.where(np.isclose(diff, 1.0))[0]

        for idx in matches:
            label = labels[idx]
            try:
                sensor, attribute = label.split(":", 1)
            except ValueError:
                continue

            existing_effect = self.effects.get(sensor, None)
            if existing_effect != attribute:
                self.effects[sensor] = attribute
                self.new_effects[sensor] = attribute
                self.new_effects_data[sensor] = episode_obj_to_msg(episode)
                self.get_logger().info(f"Found new effect! Sensor: {sensor}, Attribute: {attribute}")

    def get_effects_callback(self, request, response:GetEffects.Response):
        """
        Callback that provides the effects that have been found.

        :param request: Empty request.
        :type request: cognitive_node_interfaces.srv.GetEffects.Request
        :param response: Sensors and attributes for which effects have been found.
        :type response: cognitive_node_interfaces.srv.GetEffects.Response
        :return: Sensors and attributes for which effects have been found.
        :rtype: cognitive_node_interfaces.srv.GetEffects.Response
        """        
        sensors=[]
        attributes=[]
        data=[]
        if self.new_effects:
            effects=copy(self.new_effects)
            for sensor in effects:
                sensors.append(sensor)
                attributes.append(self.new_effects.pop(sensor))
                data.append(self.new_effects_data.pop(sensor))
        response.sensors=sensors
        response.attributes=attributes
        response.effects_episodes = data
        return response

    def evaluate(self, perception=None):
        """
        Calculates drive evaluation. If any new effect has been found, drive evaluation is 1.0.

        :return: Drive evaluation and its timestamp.
        :rtype: cognitive_node_interfaces.Evaluation
        """   
        self.evaluation.evaluation=1.0 if self.new_effects else 0.0
        self.evaluation.timestamp = self.get_clock().now().to_msg()
        return self.evaluation

class PolicyEffectanceInternal(Policy, PNodeSuccess):
    """
    Policy that creates a goal that aims to reach a consolidated P-Node. 

    This class inherits from the general Policy class and PNodeSuccess class, which provides helper methods to subscribe to P-Nodes success rate.
    """    
    def __init__(self, name='policy_effectance', class_name='cognitive_nodes.policy.Policy', ltm_id=None, goal_class=None, confidence=0.5, threshold_delta=0.2, limit_depth=False, **params):
        """Constructor of the PolicyEffectanceInternal class.

        :param name: Name of the node.
        :type name: str
        :param class_name: The name of the base Policy class, defaults to 'cognitive_nodes.policy.Policy'.
        :type class_name: str
        :param ltm_id: Id of the LTM that includes the nodes.
        :type ltm_id: str
        :param goal_class: Class of the goal to be created.
        :type goal_class: str
        :param confidence: Minimum confidence of a P-Node that allows a goal to be created, defaults to 0.5.
        :type confidence: float
        :param threshold_delta: Parameter passed to the created goal, defaults to 0.2.
        :type threshold_delta: float,
        :param limit_depth: If true, excludes from analysis P-Nodes that have resulted from a sub-goal related to another P-Node.
        :type limit_depth: bool
        :raises Exception: Raises an exception if no LTM name is provided.
        """        
        super().__init__(name, class_name, **params)
        if ltm_id is None:
            raise Exception('No LTM input was provided.')
        else:    
            self.LTM_id = ltm_id
        self.limit_depth=limit_depth
        if self.limit_depth:
            self.get_logger().error("DEBUG MESSAGE - DEPTH LIMIT ACTIVE")
        self.confidence=confidence
        self.threshold_delta=threshold_delta
        self.goal_class=goal_class
        self.index=0
        self.configure_pnode_success(self.LTM_id, self.cbgroup_client)
        self.pnode_goals_dict={}

    #UGLY HACK: This was done to limit effectance chains to a depth of 1. 
    # This must be done properly by analyzing neighbor chains and be general for any desired depth
    def pnode_success_callback(self, msg):
        """
        Callback that proccesses a success message from a P-Node.

        :param msg: Message with success information.
        :type msg: cognitive_node_interfaces.msg.SuccessRate
        """    

        if not self.limit_depth:
            return super().pnode_success_callback(msg)
        else:
            pnode = msg.node_name
            goal_linked = msg.flag
            success_rate = msg.success_rate
            if "reach_pnode_" in pnode:
                self.pnode_evaluation[pnode] = 0.0
            else:
                self.pnode_evaluation[pnode] = success_rate * (not goal_linked)

    async def process_effectance(self):
        """
        This method proccesses the effectance policy. Selects the higher confidence P-Node.
        and creates a goal linked to it if the confidence threshold is exceeded.
        """
        pnode=self.select_pnode()
        if self.pnode_evaluation[pnode] > self.confidence:
            await self.create_goal(pnode)
        else:
            self.get_logger().info("No PNode is elegible for creating a goal.")

    def select_pnode(self):
        """
        Selects the P-Node with the highest confidence.
        """
        self.get_logger().info(f"DEBUG: PNode Success Rates: {self.pnode_evaluation}")
        return max(zip(self.pnode_evaluation.values(), self.pnode_evaluation.keys()))[1]

    def find_goals(self, ltm_dump):
        """
        Creates a dictionary with the P-Nodes as keys and a list of the upstream goals as values.

        :param ltm_dump: Dictionary with the data from the LTM.
        :type ltm_dump: dict
        :return: P-Node-Goal dictionary.
        :rtype: dict
        """        
        pnodes = ltm_dump["PNode"]
        cnode_list = ltm_dump["CNode"]
        cnodes = {}
        goals = {}

        #Get the C-Node that corresponds to each P-Node
        for cnode in cnode_list:
            cnode_neighbors = cnode_list[cnode]['neighbors']
            pnode= next((node["name"] for node in cnode_neighbors if node["node_type"] == "PNode"), None)
            if pnode is not None:
                cnodes[pnode] = cnode

        for pnode, cnode in cnodes.items(): 
            cnode_neighbors = ltm_dump["CNode"][cnode]["neighbors"]
            goals[pnode] = [node["name"] for node in cnode_neighbors if node["node_type"] == "Goal"]
        self.get_logger().info(f"DEBUG: {goals}")
        return goals
        
    def changes_in_pnodes(self, ltm_dump):
        """
        Returns True if a P-Node has been added or deleted.

        :param ltm_dump: Dictionary with the data from the LTM.
        :type ltm_dump: dict
        :return: Changes in P-Nodes.
        :rtype: bool
        """
        current_pnodes = set(self.pnode_goals_dict.keys())
        new_pnodes = set(ltm_dump["PNode"].keys())
        return not current_pnodes == new_pnodes
        
    async def create_goal(self, pnode_name):
        """
        Method that creates the Goal linked to a P-Node and registers it in the LTM.

        :param pnode_name: P-Node related to goal.
        :type pnode_name: str
        """        
        self.get_logger().info(f"Creating goal linked to P-Node: {pnode_name}...")
        goal_name = f"reach_pnode_{self.index}"
        self.index+=1
        goals = self.pnode_goals_dict[pnode_name]
        self.get_logger().info(f"DEBUG: Goals Dict: {goals}")

        neighbor_dict = {pnode_name: "PNode"} 
        for goal in goals:
            neighbor_dict[goal]="Goal"

        neighbors = {
            "neighbors": [{"name": node, "node_type": node_type} for node, node_type in neighbor_dict.items()]
        }
        limits= {"threshold_delta": self.threshold_delta}
        params={**neighbors, **limits}
        self.get_logger().info(f"DEBUG: Neighbor list: {neighbors}")
        goal_response = await self.create_node_client(name=goal_name, class_name=self.goal_class, parameters=params)
        pnode_response = await self.add_neighbor_client(pnode_name, goal_name)
        if not goal_response.created or not pnode_response.success:
            self.get_logger().fatal(f"Failed creation of Goal {goal_name}")

    def read_ltm(self, ltm_dump):
        """
        Extracts information from the data provided by the LTM.

        :param ltm_dump: Dictionary with the data from the LTM.
        :type ltm_dump: dict
        """        
        super().read_ltm(ltm_dump)
        changes = self.changes_in_pnodes(ltm_dump)
        if changes:
            self.pnode_goals_dict = self.find_goals(ltm_dump)
    
    async def execute_callback(self, request, response):
        """
        Callback that executes the policy.

        :param request: Execution request.
        :type request: cognitive_node_interfaces.srv.Execute.Request
        :param response: Execution response.
        :type response: cognitive_node_interfaces.srv.Execute.Response
        :return: Execution response.
        :rtype: cognitive_node_interfaces.srv.Execute.Response
        """        
        self.get_logger().info('Executing policy: ' + self.name + '...')
        await self.process_effectance()
        response.policy=self.name
        return response

class PolicyEffectanceExternal(Policy):
    """
    Policy that creates a goal that aims to recreate an effect in the environment. 

    This class inherits from the general Policy class.
    """  
    def __init__(self, name='policy', class_name='cognitive_nodes.policy.Policy', drive_name=None, ltm_id=None, goal_class=None, goal_params={}, space_class=None, space_params={}, pnode_class=None, pnode_params={}, **params):
        """
        Constructor of the PolicyEffectanceExternal class.

        :param name: Name of the node.
        :type name: str
        :param class_name: The name of the base Policy class, defaults to 'cognitive_nodes.policy.Policy'.
        :type class_name: str
        :param drive_name: Name of the related DriveEffectanceExternal.
        :type drive_name: str
        :param ltm_id: Id of the LTM that includes the nodes.
        :type ltm_id: str
        :param goal_class: Class of the goal to be created.
        :type goal_class: str
        :param space_class: Class of the space that will be passed to the Goal.
        :type space_class: str
        :raises Exception: Raises an exception if no LTM name is provided.
        :raises RuntimeError: Raises an exception if no effects drive name is provided.
        :raises RuntimeError: Raises an exception if no goal class name is provided.
        :raises RuntimeError: Raises an exception if no space class name is provided.
        """        
        super().__init__(name, class_name, **params)
        if ltm_id is None:
            raise Exception('No LTM input was provided.')
        else:    
            self.LTM_id = ltm_id
        if drive_name is None:
            raise RuntimeError('No effects drive was provided.')
        else:    
            self.drive = drive_name
        if goal_class is None:
            raise RuntimeError('No goal class was provided.')
        else:    
            self.goal_class = goal_class
            self.goal_params = goal_params
        self.setup_connectors() # Sets up the default classes for the cognitive nodes based on the Connectors attribute if it exists.

        # Setup the space class and parameters.
        # TODO: A custom space must be developed that represents the external effects. Currently, a custom reward function is used to provide reward when the effect is recreated.
        #       For compatibility with other consolidation processes, a space is learned from rewarded points. This is not optimal, since the space can be represented by the effect itself, but it is a temporary solution. 
        if space_class is None:
            self.get_logger().warning('No space class was provided. Using default space class defined in the connectors.')
            self.space_class = self.default_class.get("Space", None)
            self.space_params = self.default_params.get("Space", {})
            if self.space_class is None:
                raise RuntimeError('No default space class was defined in the parameters nor in the connectors. Please provide a space class.')
        else:    
            self.space_class = space_class
            self.space_params = space_params

        # Get the P-Node class and parameters.
        if pnode_class is None:
            self.get_logger().warning('No P-Node class was provided. Using default P-Node class defined in the connectors.')
            self.pnode_class = self.default_class.get("PNode", None)
            self.pnode_params = self.default_params.get("PNode", {})
            if self.pnode_class is None:
                raise RuntimeError('No default P-Node class was defined in the parameters nor in the connectors. Please provide a P-Node class.')
        else:    
            self.pnode_class = pnode_class
            self.pnode_params = pnode_params
        
        self.cnode_class = self.default_class.get("CNode", None)
        self.cnode_params = self.default_params.get("CNode", {})
        if self.cnode_class is None:
            raise RuntimeError('No default C-Node class was defined in the parameters nor in the connectors. Please provide a C-Node class.')
        self.effects_client = ServiceClientAsync(self, GetEffects, f"drive/{self.drive}/get_effects", callback_group=self.cbgroup_client)

    async def execute_callback(self, request, response):
        """
        Callback that executes the policy.

        :param request: Execution request.
        :type request: cognitive_node_interfaces.srv.Execute.Request
        :param response: Execution response.
        :type response: cognitive_node_interfaces.srv.Execute.Response
        :return: Execution response.
        :rtype: cognitive_node_interfaces.srv.Execute.Response
        """
        self.get_logger().info('Executing policy: ' + self.name + '...')
        effects_msg = await self.effects_client.send_request_async()
        for sensor, attribute, episode_msg in zip(effects_msg.sensors, effects_msg.attributes, effects_msg.effects_episodes):
            episode = container_msg_to_episode(episode_msg)
            await self.create_cnode(sensor, attribute, episode)
        response.policy=self.name
        return response
    
    async def log_cnode_execution(self, cnode_name, success):
        """
        This method logs the execution of a C-Node.

        :param cnode_name: Name of the C-Node that was executed.
        :type cnode_name: str
        :param success: Boolean indicating whether the execution was successful.
        :type success: bool
        """
        logging_service = f"cnode/{cnode_name}/log_execution"
        if logging_service not in self.node_clients:
            self.node_clients[logging_service] = ServiceClientAsync(self, LogExecution, logging_service, self.cbgroup_client)
        response = await self.node_clients[logging_service].send_request_async(success=success)
        return response.added
    
    async def create_goal(self, sensor, attribute, point):
        """
        Method that creates a goal related to an effect and registers it in the LTM.

        :param sensor: Name of the sensor to which the effect is related.
        :type sensor: str
        :param attribute: Attribute in the sensor to which the effect is related.
        :type attribute: str
        """        
        self.get_logger().info(f"Creating goal linked to effect in sensor {sensor}, attribute {attribute}")
        goal_name=f"effect_{sensor}_{attribute}"
        params=dict(sensor=sensor, attribute=attribute, space_class=self.space_class, space_parameters=self.space_params, **self.goal_params)
        goal_response = await self.create_node_client(name=goal_name, class_name=self.goal_class, parameters=params)
        if not goal_response.created:
            raise RuntimeError(f"Failed creation of Goal {goal_name}")
        service_name = f"goal/{goal_name}/add_points"
        if service_name not in self.node_clients:
            self.node_clients[service_name] = ServiceClientAsync(self, AddPoints, service_name, self.cbgroup_client)

        perception_msg = point.to_msg()
        response = await self.node_clients[service_name].send_request_async(points=perception_msg, confidences=[1.0])
        if not response.added:
            raise RuntimeError(f"Failed to add point to Goal {goal_name}")
        return goal_name
    
    async def create_pnode(self, ident, point):
        """
        Method that creates a P-Node related to an effect and registers it in the LTM.

        :param ident: Identifier for the P-Node.
        :type ident: str
        :param point: Point in the space that represents the effect.
        :type point: core.container.Container
        """
        pnode_name=f"pnode_{ident}"
        pnode_params = self.pnode_params
        response = await self.create_node_client(name=pnode_name, class_name=self.pnode_class, parameters=pnode_params)
        if not response.created:
            raise RuntimeError(f"Failed creation of P-Node {pnode_name}")
        service_name = f"pnode/{pnode_name}/add_points"
        if service_name not in self.node_clients:
            self.node_clients[service_name] = ServiceClientAsync(self, AddPoints, service_name, self.cbgroup_client)

        perception_msg = point.to_msg()
        response = await self.node_clients[service_name].send_request_async(points=perception_msg, confidences=[1.0])
        if not response.added:
            raise RuntimeError(f"Failed to add point to P-Node {pnode_name}")

        return pnode_name

    async def create_cnode(self, sensor, attribute, episode):
        """
        Method that creates all the nodes that correspond to the 

        :param sensor: Name of the sensor to which the effect is related.
        :type sensor: str
        :param attribute: Attribute in the sensor to which the effect is related.
        :type attribute: str
        :param episode: Episode that generated the effect.
        :type episode: cognitive_nodes.episode.Episode
        """
        # Load the parameters for the C-Node, P-Node, and Goal from the connectors or defaults.
        cnode_class = self.cnode_class
        cnode_params = self.cnode_params
        world_model = episode.domain_name
        policy = episode.parent_policy

        # Create the goal and P-Node, and set up their relationships.
        goal = await self.create_goal(sensor, attribute, episode.perception)
        ident = f"{world_model}__{goal}__{policy}"
        pnode_name = await self.create_pnode(ident, episode.old_perception)
        neighbor_dict = {world_model: "WorldModel", pnode_name: "PNode", goal: "Goal"}
        neighbors = {
            "neighbors": [{"name": node, "node_type": node_type} for node, node_type in neighbor_dict.items()]
        }

        # Create the C-Node with the specified parameters and neighbors.
        cnode_name = f"cnode_{ident}"
        cnode = await self.create_node_client(
            name=cnode_name, class_name=cnode_class, parameters={**cnode_params, **neighbors}
        )
        if not cnode.created:
            raise RuntimeError(f"Failed creation of C-Node {cnode_name}")

        #Add new C-Node as neighbor of the corresponding policy
        updated_policy = await self.update_neighbor_client(policy, cnode_name, operation=True)
        if not updated_policy.success:
            raise RuntimeError(f"Failed to add C-Node {cnode_name} as neighbor of policy {policy}")
        await self.log_cnode_execution(cnode_name, success=True)

class GoalActivatePNode(GoalLearnedSpace):
    """
    Goal that provides reward when the related P-Nodes goes from not activated to activated.

    This class inherits from the GoalLearnedSpace class.
    """    
    def __init__(self, name='goal', class_name='cognitive_nodes.goal.Goal', threshold_delta=0.2, **params):
        """
        Constructor of the GoalActivatePNode class.

        :param name: Name of the node.
        :type name: str
        :param class_name: The name of the base Goal class, defaults to 'cognitive_nodes.goal.Goal'.
        :type class_name: str
        :param threshold_delta: Minimum change in activation that triggers a reward, defaults to 0.2.
        :type threshold_delta: float
        """        
        super().__init__(name, class_name, **params)
        self.threshold_delta=threshold_delta
        self.setup_pnode()
    
    def setup_pnode(self):
        """
        Creates the required service clients and subscriptions.
        """        
        pnode = next((node["name"] for node in self.neighbors if node["node_type"] == "PNode"))
        self.pnode_activation_client = ServiceClientAsync(self, GetActivation, f"cognitive_node/{pnode}/get_activation", self.cbgroup_client)
        self.pnode_space_client = ServiceClientAsync(self, SendSpace, f"pnode/{pnode}/send_space", self.cbgroup_client) 
        self.pnode_contains_client = ServiceClientAsync(self, ContainsSpace, f"pnode/{pnode}/contains_space", self.cbgroup_client) 
        self.pnode_confidence = self.create_subscription(SuccessRate, f'pnode/{str(pnode)}/success_rate', self.read_confidence, 1, callback_group=self.cbgroup_activation) #TODO: REMOVE?

    def calculate_reward(self, drive_name = None):
        """
        This goal does not take into account a drive to obtain reward. This method overrides the default behavior.
        """        
        return None
    
    def read_confidence(self, msg:SuccessRate): #TODO: REMOVE?
        """
        Reads the confidence value from the SuccessRate message and updates the confidence attribute.

        :param msg: Message containing the success rate of the P-Node.
        :type msg: SuccessRate
        """
        self.confidence = msg.success_rate
        self.confidence=msg.success_rate

    async def send_goal_space_callback(self, request, response):
        """
        This method overrides the default behavior of the send space service. Obtains the space from the P-Node and sends it as response.

        :param request: Empty request.
        :type request: cognitive_node_interfaces.srv.SendSpace.Request
        :param response: Space data.
        :type response: cognitive_node_interfaces.srv.SendSpace.Response
        :return: Space data.
        :rtype: cognitive_node_interfaces.srv.SendSpace.Response
        """        
        response = await self.pnode_space_client.send_request_async()
        return response
    
    async def contains_space_callback(self, request, response):
        """
        This method overrides the default behavior of the contains space service. Obtains checks if the space is contained in the P-Node and sends it as response.

        :param request: Data of the space.
        :type request: cognitive_node_interfaces.srv.ContainsSpace.Request
        :param response: Boolean that indicates if the space is contained inside the goal.
        :type response: cognitive_node_interfaces.srv.ContainsSpace.Response
        :return: Boolean that indicates if the space is contained inside the goal.
        :rtype: cognitive_node_interfaces.srv.ContainsSpace.Response
        """        
        response = await self.pnode_contains_client.send_request_async(space=request.space)
        return response

    async def get_reward(self, old_perception=None, perception=None):
        """
        Method that obtains the reward for the goal. It recieves two consecutive perceptions, calculates the related P-Node activation for each and detects if the P-Node was activated.

        :param old_perception: First state perception dictionary.
        :type old_perception: core.container.Container
        :param perception: Second state perception dictionary.
        :type perception: core.container.Container
        :return: Reward and current timestamp.
        :rtype: Tuple (float, builtin_interfaces.msg.Time)
        """        
        old_perception_msg = old_perception.to_msg() if old_perception else None
        perception_msg = perception.to_msg() if perception else None
        old_activation = (await self.pnode_activation_client.send_request_async(perception=old_perception_msg)).activation[0]
        activation = (await self.pnode_activation_client.send_request_async(perception=perception_msg)).activation[0]
        if activation-old_activation>self.threshold_delta:
            self.reward = 1.0
        else:
            self.reward = 0.0
        self.publish_success_rate()
        return self.reward, self.get_clock().now().to_msg() 
    
class GoalRecreateEffect(GoalLearnedSpace):
    """
    Goal that provides reward when the related P-Nodes goes from not activated to activated.

    This class inherits from the GoalLearnedSpace class.
    """    
    def __init__(self, name='goal', class_name='cognitive_nodes.goal.Goal', sensor=None, attribute=None, **params):
        """
        Constructor of the GoalRecreateEffect class.

        :param name: Name of the node.
        :type name: str
        :param class_name: The name of the base Goal class, defaults to 'cognitive_nodes.policy.Policy'.
        :type class_name: str
        :param sensor: Name of the related sensor.
        :type sensor: str
        :param attribute: Name of the related attribute of the sensor.
        :type attribute: str
        :raises Exception: Raises exeption if the sensor or the attribute are missing.
        """        
        super().__init__(name, class_name, sensor=sensor, attribute=attribute, **params)
        if not sensor or not attribute:
            raise Exception("Effect not configured")
        else:
            self.sensor=sensor
            self.attribute=attribute

    def get_reward(self, old_perception, perception):
        """
        Method that obtains reward for the goal. It recieves two consecutive perceptions, checks if the effect was recreated (the related attribute went from 0 to 1).

        :param old_perception: First state perception dictionary.
        :type old_perception: core.container.Container
        :param perception: Second state perception dictionary
        :type perception: core.container.Container
        :return: Reward and current timestamp.
        :rtype: Tuple (float, builtin_interfaces.msg.Time)
        """        
        if not compare_perceptions(old_perception, perception):
            expected_reward=self.get_expected_reward(perception)
            effect, old_sensing, _= self.process_effect(old_perception, perception)
            if old_sensing<1.0:
                reward=float(effect)
                self.update_space(reward, expected_reward, perception)
                self.get_logger().info(f"DEBUG - GOAL: {self.name} REWARD: {reward} PRED_REWARD: {expected_reward} CONF: {self.confidence}")
                timestamp=self.get_clock().now().to_msg()
            else:
                self.get_logger().info(f"DEBUG - {self.name} - Effect already active")
                reward=0.0
                timestamp=self.get_clock().now().to_msg()
        else:
            reward=0.0
            timestamp=self.get_clock().now().to_msg()
        return reward, timestamp
       
    def process_effect(self, old_perception, perception):
        """
        Method that extracts the appropriate reading from the perceptions and returns if effect is found.
        

        :param old_perception: First state perception dictionary.
        :type old_perception: core.container.Container
        :param perception: Second state perception dictionary.
        :type perception: core.container.Container
        :return: Tuple with a boolean True if effect is found and the raw readings of the sensor's attribute for both perceptions.
        :rtype: tuple (bool, float, float)
        """        
        effect=False
        label = f"{self.sensor}:{self.attribute}"

        if not label in perception.feature_labels or not label in old_perception.feature_labels:
            return False, 0.0, 0.0

        old_sensing=old_perception.read().sel(features=label).values[-1]
        sensing=perception.read().sel(features=label).values[-1]
        effect=isclose(sensing-old_sensing, 1.0)
        
        return effect, old_sensing, sensing
    
    # def calculate_activation(self, perception, activation_list):
    #     """
    #     This method extends the default calculate activation method for goals and provides activation based on the goal's confidence.

    #     :param perception: The perception for which the activation will be calculated. None can be passed.
    #     :type perception: dict
    #     :param activation_list: List of activations considered in the node.
    #     :type activation_list: dict
    #     """        
    #     #Calculates activation as any goal
    #     super().calculate_activation(perception, activation_list)
    #     #Provides activation if not learned depending on confidence
    #     if not self.learned_space:
    #         self.activation.activation=max((1 - self.confidence) * 0.5 + 0.5, self.activation.activation)

