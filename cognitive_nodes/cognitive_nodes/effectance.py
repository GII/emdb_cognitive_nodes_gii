import yaml
import numpy as np

from collections import deque

from cognitive_nodes.need import Need
from cognitive_nodes.drive import Drive
from cognitive_nodes.goal import Goal, GoalMotiven
from cognitive_nodes.policy import Policy
from core.service_client import ServiceClient, ServiceClientAsync

from std_msgs.msg import String
from core_interfaces.srv import GetNodeFromLTM, CreateNode
from cognitive_node_interfaces.msg import SuccessRate
from cognitive_node_interfaces.srv import GetActivation
from core.utils import perception_dict_to_msg

class LTMSubscription:
    def configure_ltm_subscription(self, ltm):
        self.ltm_suscription = self.create_subscription(String, "state", self.ltm_change_callback, 0, callback_group=self.cbgroup_client)
        
    def ltm_change_callback(self, msg):
        self.get_logger().info("Processing change from LTM...")
        ltm_dump = yaml.safe_load(msg.data)
        self.read_ltm(ltm_dump=ltm_dump)

    def read_ltm(self, ltm_dump):
        raise NotImplementedError
    
class PNodeSuccess(LTMSubscription):
    def configure_pnode_success(self, ltm):
        self.configure_ltm_subscription(ltm)
        self.pnode_subscriptions = {}
        self.pnode_evaluation={}

    def read_ltm(self, ltm_dump):
        pnodes = ltm_dump["PNode"]
        for pnode in pnodes:
            if pnode not in self.pnode_subscriptions.keys():
                self.pnode_subscriptions[pnode] = self.create_subscription(SuccessRate, f"/pnode/{pnode}/success_rate", self.pnode_success_callback, 1, callback_group=self.cbgroup_activation)

    def pnode_success_callback(self, msg: SuccessRate):
        pnode = msg.node_name
        goal_linked = msg.flag
        success_rate = msg.success_rate
        self.pnode_evaluation[pnode] = success_rate * (not goal_linked)

class DriveEffectance(Drive, PNodeSuccess):
    #This class implements a simple effectance drive. It create goals to reach learned P-Nodes, in other words, reaching the effect of activaction of a P-Node.
    #The name might change when other types of effectances are implemented (One class might create each type or all might be included here, TBD)
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

class GoalEffectance(GoalMotiven):
    def __init__(self, name='goal', class_name='cognitive_nodes.goal.Goal', **params):
        super().__init__(name, class_name, **params)
    
    def calculate_reward(self, drive_name): #No reward is provided
        self.reward = 0.0
        return self.reward, self.get_clock().now().to_msg()

class PolicyEffectance(Policy, PNodeSuccess):
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
        pnode=self.select_pnode()
        if self.pnode_evaluation[pnode] > self.confidence:
            await self.create_goal(pnode)
        else:
            self.get_logger().info("No PNode is elegible for creating a goal")

    def select_pnode(self):
        self.get_logger().info(f"DEBUG: PNode Success Rates: {self.pnode_evaluation}")
        return max(zip(self.pnode_evaluation.values(), self.pnode_evaluation.keys()))[1]

    def find_drives(self, ltm_dump):
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
        pnodes = ltm_dump["PNode"]
        cnodes = {}
        goals = {}
        for pnode in pnodes:
            pnode_neighbors = pnodes[pnode]["neighbors"]
            cnode = next((node["name"] for node in pnode_neighbors if node["node_type"] == "CNode"), None)
            if cnode is not None:
                cnodes[pnode] = cnode
        for pnode, cnode in cnodes.items(): 
            cnode_neighbors = ltm_dump["CNode"][cnode]["neighbors"]
            goals[pnode] = [node["name"] for node in cnode_neighbors if node["node_type"] == "Goal"]
        self.get_logger().info(f"DEBUG: {goals}")
        return goals
        
    def changes_in_pnodes(self, ltm_dump):
        current_pnodes = set(self.pnode_goals_dict.keys())
        new_pnodes = set(ltm_dump["PNode"].keys())
        if current_pnodes == new_pnodes:
            return False
        else:
            added = new_pnodes - current_pnodes
            deleted = current_pnodes - new_pnodes
            if deleted:
                for node in deleted:
                    del self.pnode_goals_dict[node]          
            return True
        
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
        goal = await self.create_node_client(name=goal_name, class_name=self.goal_class, parameters=params)

        if not goal.created:
            self.get_logger().fatal(f"Failed creation of Goal {goal_name}")
        

    def create_node_client(self, name, class_name, parameters={}):
        """
        This method calls the add node service of the commander.

        :param name: Name of the node to be created.
        :type name: str
        :param class_name: Name of the class to be used for the creation of the node.
        :type class_name: str
        :param parameters: Optional parameters that can be passed to the node, defaults to {}
        :type parameters: dict, optional
        :return: Success status received from the commander
        :rtype: bool
        """

        self.get_logger().info("Requesting node creation")
        params_str = yaml.dump(parameters)
        service_name = "commander/create"
        if service_name not in self.node_clients:
            self.node_clients[service_name] = ServiceClientAsync(self, CreateNode, service_name, self.cbgroup_client)
        response = self.node_clients[service_name].send_request_async(
            name=name, class_name=class_name, parameters=params_str
        )
        return response

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

class GoalActivatePNode(GoalMotiven):
    def __init__(self, name='goal', class_name='cognitive_nodes.goal.Goal', threshold_delta=0.2, **params):
        super().__init__(name, class_name, **params)
        self.threshold_delta=threshold_delta
        self.setup_pnode()
    
    def setup_pnode(self):
        pnode = next((node["name"] for node in self.neighbors if node["node_type"] == "PNode"))
        self.pnode_activation_client = ServiceClientAsync(self, GetActivation, f"cognitive_node/{pnode}/get_activation", self.cbgroup_client)

    def calculate_reward(self, drive_name):
        return None

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


