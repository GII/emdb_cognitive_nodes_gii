import yaml
import numpy as np

from math import isclose
from copy import copy

from cognitive_nodes.need import Need
from cognitive_nodes.drive import Drive
from cognitive_nodes.goal import Goal, GoalMotiven, GoalLearnedSpace
from cognitive_nodes.policy import Policy
from core.service_client import ServiceClient, ServiceClientAsync

from std_msgs.msg import String
from core_interfaces.srv import GetNodeFromLTM, CreateNode, UpdateNeighbor
from cognitive_node_interfaces.msg import SuccessRate
from cognitive_node_interfaces.srv import GetActivation, SendSpace, GetEffects
from core.utils import perception_dict_to_msg, perception_msg_to_dict, compare_perceptions
from cognitive_nodes.utils import PNodeSuccess, EpisodeSubscription



class DriveEffectanceInternal(Drive, PNodeSuccess):
    #This class implements a simple effectance drive. It create goals to reach learned P-Nodes, in other words, reaching the effect of activaction of a P-Node.
    def __init__(self, name="drive_effectance", class_name="cognitive_nodes.drive.Drive", ltm_id=None, min_confidence=0.1, **params):
        super().__init__(name, class_name, **params)
        if ltm_id is None:
            raise Exception('No LTM input was provided.')
        else:    
            self.LTM_id = ltm_id
        self.min_confidence=min_confidence
        self.configure_pnode_success(self.LTM_id)

    def evaluate(self):
        max_pnode= max(self.pnode_evaluation.values(), default=0.0)
        if max_pnode>=self.min_confidence:
            self.evaluation.evaluation = 1.0
        else:
            self.evaluation.evaluation = 0.0
        self.evaluation.timestamp = self.get_clock().now().to_msg()
        return self.evaluation
    
class DriveEffectanceExternal(Drive, EpisodeSubscription):
    #Drive that detects external effects. This is, changes from 0 to 1 in a sensor.
    def __init__(self, name="drive_effectance", class_name="cognitive_nodes.drive.Drive", episodes_topic=None, episodes_msg=None, **params):
        super().__init__(name, class_name, **params)
        if not episodes_topic or not episodes_msg:
            raise Exception('No LTM input was provided.')
        else:    
            self.episode_topic = episodes_topic
            self.episode_msg = episodes_msg
        self.effects={} 
        self.new_effects={}
        self.get_effects_service = self.create_service(GetEffects, 'drive/' + str(
            name) + '/get_effects', self.get_effects_callback, callback_group=self.cbgroup_server)
        self.configure_episode_subscription(episodes_topic, episodes_msg)
    
    def episode_callback(self, msg):
        perception=perception_msg_to_dict(msg.perception)
        old_perception=perception_msg_to_dict(msg.old_perception)
        self.find_effects(perception, old_perception)

    def find_effects(self, perception, old_perception):
        for sensor, data in perception.items():
            for index, object in enumerate(data):
                for attribute, _ in object.items():
                    sensing=perception[sensor][index][attribute]
                    old_sensing=old_perception[sensor][index][attribute]
                    if isclose(sensing-old_sensing, 1.0):
                        existing_effect=self.effects.get(sensor, None)
                        if existing_effect!=attribute:
                            self.effects[sensor]=attribute
                            self.new_effects[sensor]=attribute
                            self.get_logger().info(f"Found new effect! Sensor: {sensor}, Attribute: {attribute}")

    def get_effects_callback(self, request, response:GetEffects.Response):
        sensors=[]
        attributes=[]
        if self.new_effects:
            effects=copy(self.new_effects)
            for sensor in effects:
                sensors.append(sensor)
                attributes.append(self.new_effects.pop(sensor))
        response.sensors=sensors
        response.attributes=attributes
        return response

    def evaluate(self, perception=None):
        self.evaluation.evaluation=1.0 if self.new_effects else 0.0
        self.evaluation.timestamp = self.get_clock().now().to_msg()
        return self.evaluation

class PolicyEffectanceInternal(Policy, PNodeSuccess):
    def __init__(self, name='policy_effectance', class_name='cognitive_nodes.policy.Policy', ltm_id=None, goal_class=None, confidence=0.5, threshold_delta=0.2, **params):
        super().__init__(name, class_name, **params)
        if ltm_id is None:
            raise Exception('No LTM input was provided.')
        else:    
            self.LTM_id = ltm_id
        self.confidence=confidence
        self.threshold_delta=threshold_delta
        self.goal_class=goal_class
        self.index=0
        self.configure_pnode_success(self.LTM_id)
        self.pnode_goals_dict={}

    async def process_effectance(self):
        """
        This method proccesses the effectance policy. Selects the higher confidence P-Node
        and creates a goal linked to it if the confidence threshold is exceeded.
        """
        pnode=self.select_pnode()
        if self.pnode_evaluation[pnode] > self.confidence:
            await self.create_goal(pnode)
        else:
            self.get_logger().info("No PNode is elegible for creating a goal")

    def select_pnode(self):
        """
        Selects the P-Node with the highest confidence
        """
        self.get_logger().info(f"DEBUG: PNode Success Rates: {self.pnode_evaluation}")
        return max(zip(self.pnode_evaluation.values(), self.pnode_evaluation.keys()))[1]

    def find_drives(self, ltm_dump):
        """
        ----DEPRECATED----
        Searches the drives upstream from the P-Nodes
        """
        pnodes = ltm_dump["PNode"]
        cnodes = {}
        goals = {}
        drives = {}
        for pnode in pnodes:
            pnode_neighbors = pnodes[pnode]["neighbors"]
            cnode = next((node["name"] for node in pnode_neighbors if node["node_type"] == "CNode"), None)
            if cnode is not None:
                cnodes[pnode] = cnode
        for pnode, cnode in cnodes.items(): 
            cnode_neighbors = ltm_dump["CNode"][cnode]["neighbors"]
            goals[pnode] = next((node["name"] for node in cnode_neighbors if node["node_type"] == "Goal"))
        self.get_logger().info(f"DEBUG: {goals}")
        for pnode, goal in goals.items():
            goal_neighbors = ltm_dump["Goal"][goal]["neighbors"]
            drives[pnode] = [node["name"] for node in goal_neighbors if node["node_type"] == "Drive"]
        return drives
    
    def find_goals(self, ltm_dump):
        """
        Creates a dictionary with the P-Nodes as keys and a list of the upstream goals as values
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
        Returns True if a P-Node has been added or deleted
        """
        current_pnodes = set(self.pnode_goals_dict.keys())
        new_pnodes = set(ltm_dump["PNode"].keys())
        return not current_pnodes == new_pnodes
        
    async def create_goal(self, pnode_name):
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
        super().read_ltm(ltm_dump)
        changes = self.changes_in_pnodes(ltm_dump)
        if changes:
            self.pnode_goals_dict = self.find_goals(ltm_dump)
    
    async def execute_callback(self, request, response):
        self.get_logger().info('Executing policy: ' + self.name + '...')
        await self.process_effectance()
        response.policy=self.name
        return response

class PolicyEffectanceExternal(Policy):
    def __init__(self, name='policy', class_name='cognitive_nodes.policy.Policy', drive_name=None, ltm_id=None, goal_class=None, space_class=None, **params):
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
        if space_class is None:
            raise RuntimeError('No space class for the goal was provided.')
        else:    
            self.space_class = space_class
            
        self.effects_client = ServiceClientAsync(self, GetEffects, f"drive/{self.drive}/get_effects", callback_group=self.cbgroup_client)
    
    async def execute_callback(self, request, response):
        self.get_logger().info('Executing policy: ' + self.name + '...')
        effects_msg = await self.effects_client.send_request_async()
        for sensor, attribute in zip(effects_msg.sensors, effects_msg.attributes):
            await self.create_goal(sensor, attribute)
        response.policy=self.name
        return response
    
    async def create_goal(self, sensor, attribute):
        self.get_logger().info(f"Creating goal linked to effect in sensor {sensor}, attribute {attribute}")
        goal_name=f"effect_{sensor}_{attribute}"
        params=dict(sensor=sensor, attribute=attribute, space_class=self.space_class, history_size=300, min_confidence=0.95)
        goal_response = await self.create_node_client(name=goal_name, class_name=self.goal_class, parameters=params)
        if not goal_response.created:
            self.get_logger().fatal(f"Failed creation of Goal {goal_name}")

class GoalActivatePNode(GoalMotiven):
    def __init__(self, name='goal', class_name='cognitive_nodes.goal.Goal', threshold_delta=0.2, **params):
        super().__init__(name, class_name, **params)
        self.threshold_delta=threshold_delta
        self.send_goal_space_service = self.create_service(SendSpace, 'goal/' + str(
            name) + '/send_space', self.send_goal_space_callback, callback_group=self.cbgroup_server)
        self.setup_pnode()
    
    def setup_pnode(self):
        pnode = next((node["name"] for node in self.neighbors if node["node_type"] == "PNode"))
        self.pnode_activation_client = ServiceClientAsync(self, GetActivation, f"cognitive_node/{pnode}/get_activation", self.cbgroup_client)
        self.pnode_space_client = ServiceClientAsync(self, SendSpace, f"pnode/{pnode}/send_space", self.cbgroup_client) 

    def calculate_reward(self, drive_name):
        return None

    async def send_goal_space_callback(self, request, response):
        response = await self.pnode_space_client.send_request_async()
        return response

    async def get_reward(self, old_perception=None, perception=None):
        old_perception_msg=perception_dict_to_msg(old_perception)
        perception_msg=perception_dict_to_msg(perception)
        old_activation = (await self.pnode_activation_client.send_request_async(perception=old_perception_msg)).activation
        activation = (await self.pnode_activation_client.send_request_async(perception=perception_msg)).activation
        if activation-old_activation>self.threshold_delta:
            self.reward = 1.0
        else:
            self.reward = 0.0
        return self.reward, self.get_clock().now().to_msg() 
    
class GoalRecreateEffect(GoalLearnedSpace):
    def __init__(self, name='goal', class_name='cognitive_nodes.goal.Goal', sensor=None, attribute=None, **params):
        super().__init__(name, class_name, **params)
        if not sensor or not attribute:
            raise Exception("Effect not configured")
        else:
            self.sensor=sensor
            self.attribute=attribute

    def get_reward(self, old_perception, perception):
        if not compare_perceptions(old_perception, perception):
            expected_reward=self.get_expected_reward(perception)
            reward=float(self.process_effect(old_perception, perception))
            self.update_space(reward, expected_reward, perception)
            self.get_logger().info(f"DEBUG - GOAL: {self.name} REWARD: {reward} PRED_REWARD: {expected_reward}")
            timestamp=self.get_clock().now().to_msg()
        else:
            reward=0.0
            timestamp=self.get_clock().now().to_msg()
        return reward, timestamp
    
    def process_effect(self, old_perception, perception):
        effect=False
        for index, _ in enumerate(perception[self.sensor]):
            old_sensing=old_perception[self.sensor][index][self.attribute]
            sensing=perception[self.sensor][index][self.attribute]
            effect=isclose(sensing-old_sensing, 1.0)
        return effect
    
    def calculate_activation(self, perception, activation_list):
        #Calculates activation regularly
        super().calculate_activation(perception, activation_list)
        #Provides activation if not learned depending on competence
        if not self.learned_space:
            self.activation.activation=max((1 - self.confidence) * 0.5 + 0.5, self.activation.activation)
        else:
            self.activation.activation=max(0.015, self.activation.activation)
        


