import yaml

from collections import deque
from numpy.random import Generator

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
        super().__init__(name, **params)
        self.policies = PolicyQueue()
        self.LTM_id = "ltm_0"
        self.ltm_subscriber = self.create_subscription(String, "/state", self.ltm_change_callback, 1, callback_group=self.cbgroup_client)
        self.setup()

    def setup(self):
        ltm = self.request_ltm()
        self.configure_policies(ltm)

    def request_ltm(self):
        # Call get_node service from LTM
        service_name = "/" + str(self.LTM_id) + "/get_node"
        request = ""
        client = ServiceClient(GetNodeFromLTM, service_name)
        ltm_response = client.send_request(name=request)
        ltm = yaml.safe_load(ltm_response.data)
        return ltm
    
    def ltm_change_callback(self, msg):
        ltm = yaml.safe_load(msg.data)
        self.configure_policies(ltm)

    def configure_policies(self, ltm_cache):
        pass

    def execute_callback(self, request, response):
        return super().execute_callback(request, response)
    

class PolicyQueue:
    def __init__(self):
        self.queue = deque()

    def select_policy(self):
        policy = self.front()
        self.queue.rotate(1)
        return policy
    
    def shuffle(self, rng: Generator):
        rng.shuffle(self.queue)

    #Default access and change methods

    def enqueue(self, x):
        return self.queue.appendleft(x)
    
    def dequeue(self):
        return self.queue.pop()
    
    def isEmpty(self):
        return len(self.queue) == 0
    
    def front(self):
        return self.queue[-1]
    
    def rear(self):
        return self.queue[0]
    
