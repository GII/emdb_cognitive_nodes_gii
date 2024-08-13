import yaml
import numpy as np

from collections import deque

from cognitive_nodes.need import Need
from cognitive_nodes.drive import Drive
from cognitive_nodes.goal import Goal
from cognitive_nodes.policy import PolicyBlocking
from core.service_client import ServiceClient, ServiceClientAsync

from std_msgs.msg import String
from core_interfaces.srv import GetNodeFromLTM


#Work In Progress
class DriveNovelty(Drive):
    def __init__(self, name="drive", class_name="cognitive_nodes.drive.Drive", **params):
        super().__init__(name, class_name, **params)

    def evaluate(self, perception=None):
        self.evaluation.evaluation = 1.0
        self.evaluation.timestamp = self.get_clock().now().to_msg()
        return self.evaluation
    
class PolicyNovelty(PolicyBlocking):
    def __init__(self, name='policy_novelty', service_msg=None, service_name=None, **params):
        super().__init__(name, service_msg=service_msg, service_name=service_name, **params)
        self.policies = PolicyQueue()
        self.LTM_id = "ltm_0"
        self.setup()
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
        self.policies.remove(self.name)
        self.get_logger().info(f"Configured policies: {self.policies.queue}")

    async def execute_callback(self, request, response):
        policy = self.policies.select_policy()
        self.get_logger().info('Executing policy: ' + policy + '...')
        await self.policy_service.send_request_async(policy=policy)
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
    
