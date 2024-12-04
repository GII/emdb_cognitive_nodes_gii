import yaml
import numpy as np
from numpy import bool_, float_

from collections import deque
from copy import deepcopy

from cognitive_nodes.need import Need
from cognitive_nodes.drive import Drive
from cognitive_nodes.goal import Goal
from cognitive_nodes.policy import Policy, PolicyBlocking
from core.service_client import ServiceClient, ServiceClientAsync
from core.utils import actuation_dict_to_msg

from std_msgs.msg import String
from core_interfaces.srv import GetNodeFromLTM
from cognitive_node_interfaces.srv import Execute


#Work In Progress
class DriveNovelty(Drive):
    def __init__(self, name="drive", class_name="cognitive_nodes.drive.Drive", **params):
        super().__init__(name, class_name, **params)

    def evaluate(self, perception=None):
        self.evaluation.evaluation = 1.0
        self.evaluation.timestamp = self.get_clock().now().to_msg()
        return self.evaluation
    
class PolicyNovelty(Policy):
    def __init__(self, name='policy_novelty', exclude_list=[], ltm_id=None, **params):
        super().__init__(name, **params)
        self.policies = PolicyQueue()
        self.LTM_id = ltm_id
        self.exclude_list=exclude_list
        self.exclude_list.append(self.name)
        self.setup()
        self.counter=0
        self.ltm_subscriber = self.create_subscription(String, "/state", self.ltm_change_callback, 1, callback_group=self.cbgroup_client)
        

    def setup(self):
        ltm = self.request_ltm()
        random_seed = getattr(self, 'random_seed', None)
        self.rng = np.random.default_rng(random_seed)
        self.configure_policies(ltm)
        self.get_logger().info("Configured Novelty Policy.")

    def request_ltm(self):
        # Call get_node service from LTM
        service_name = "/" + str(self.LTM_id) + "/get_node"
        request = ""
        client = ServiceClient(GetNodeFromLTM, service_name)
        ltm_response = client.send_request(name=request)
        ltm = yaml.safe_load(ltm_response.data)
        return ltm
    
    def ltm_change_callback(self, msg):
        self.get_logger().debug("Reading LTM Change...")
        ltm = yaml.safe_load(msg.data)
        self.configure_policies(ltm)

    def configure_policies(self, ltm_cache):
        policies = list(ltm_cache["Policy"].keys())
        self.get_logger().info(f"Configuring Policies: {policies}")
        changes = self.policies.merge(policies)
        if changes:
            self.policies.shuffle(self.rng)
            self.counter=0
        for policy in self.exclude_list:
            self.policies.remove(policy)
        self.get_logger().info(f"Configured policies: {self.policies.queue}")

    def select_policy(self):
        policy=self.policies.select_policy()
        self.counter+=1

        if self.counter%len(self.policies) == 0:
            self.get_logger().info(f"DEBUG: Shuffling Policies (Counter: {self.counter} Policies: {len(self.policies)})")
            self.policies.shuffle(self.rng)
            self.counter=0

        return policy

    async def execute_callback(self, request, response):
        policy = self.select_policy()
        if policy not in self.node_clients:
            self.node_clients[policy] = ServiceClientAsync(self, Execute, f"policy/{policy}/execute", callback_group=self.cbgroup_client)
        self.get_logger().info('Executing policy: ' + policy + '...')
        await self.node_clients[policy].send_request_async()
        response.policy = policy
        return response
    

class PolicyQueue:
    def __init__(self):
        self.queue = deque()

    def select_policy(self):
        policy = self.front()
        self.queue.rotate(1)
        return policy
    
    def shuffle(self, rng: np.random.Generator):
        rng.shuffle(self.queue)

    def find_differences(self, items):
        new = [x for x in items if x not in self.queue]
        missing = [x for x in self.queue if x not in items]
        return new, missing
    
    def merge(self, items):
        new, missing = self.find_differences(items)
        for item in new:
            self.enqueue(item)
        for item in missing:
            self.remove(item)
        if not new and not missing:
            return False
        return True

    #Default access and change methods

    def enqueue(self, item):
        return self.queue.appendleft(item)
    
    def dequeue(self):
        return self.queue.pop()
    
    def remove(self, item):
        if item in self.queue:
            self.queue.remove(item)
            return True
        return False
    
    def isEmpty(self):
        return len(self.queue) == 0
    
    def front(self):
        return self.queue[-1]
    
    def rear(self):
        return self.queue[0]
    
    def exists(self, item):
        return item in self.queue
    
    def __len__(self):
        return len(self.queue)
    

class PolicyRandomAction(PolicyBlocking):
    def __init__(self, name='policy_random_action', actuation_config=None, **params):
        super().__init__(name, **params)
        self.actuation_config=actuation_config
        self.actuation={} #All fields are normalized 0 to 1
        self.setup()

    def setup(self):
        random_seed = getattr(self, 'random_seed', None)
        self.rng = np.random.default_rng(random_seed)
        for actuator in self.actuation_config:
            self.actuation[actuator]=[{}]
            for param in self.actuation_config[actuator]:
                if self.actuation_config[actuator][param]["type"] == "float":
                    self.actuation[actuator][0][param]=0.0
                elif self.actuation_config[actuator][param]["type"] == "bool":
                    self.actuation[actuator][0][param]=False
                else:
                    raise TypeError("Type assigned to actuator not recognized")
    
    def randomize_actuation(self):

        for actuator in self.actuation:
            for param in self.actuation[actuator][0]:
                self.get_logger().info(f"DEBUG: {actuator}, {param} : {self.actuation[actuator][0][param]}")

                #self.get_logger().info(f"DEBUG: {actuator}, {param} : {self.actuation_config[actuator][param]["type"]}")
                if self.actuation_config[actuator][param]["type"]=="float":
                    self.actuation[actuator][0][param]=self.rng.uniform()
                elif self.actuation_config[actuator][param]["type"]=="bool":
                    self.actuation[actuator][0][param]=self.rng.choice([True,True,True,True,True,False])
                else:
                    self.get_logger().info(f"DEBUG: {actuator}, {param} {self.actuation[actuator][0][param]} type: {type(self.actuation[actuator][0][param])}")
                    raise TypeError("Actuation parameter is of unknown type")

    
    def denormalize_actuation(self, actuation, actuation_config):
        act=deepcopy(actuation)
        for actuator in act:
            for param in act[actuator][0]:
                if actuation_config[actuator][param]["type"]=="float":
                    bounds=actuation_config[actuator][param]["bounds"]
                    value=act[actuator][0][param]
                    act[actuator][0][param]=bounds[0]+(value*(bounds[1]-bounds[0]))
        return act


    async def execute_callback(self, request, response):
        """
        Makes a service call to the server that handles the execution of the policy.

        :param request: The request to execute the policy.
        :type request: cognitive_node_interfaces.srv.ExecutePolicy_Request
        :param response: The response indicating the executed policy.
        :type response: cognitive_node_interfaces.srv.ExecutePolicy_Response
        :return: The response with the executed policy name.
        :rtype: cognitive_node_interfaces.srv.ExecutePolicy_Response
        """
        self.get_logger().info('Executing policy: ' + self.name + '...')
        self.randomize_actuation()
        actuation_msg=actuation_dict_to_msg(self.denormalize_actuation(self.actuation, self.actuation_config))
        await self.policy_service.send_request_async(action=actuation_msg)
        response.policy=self.name
        response.action=actuation_dict_to_msg(self.actuation)
        return response
