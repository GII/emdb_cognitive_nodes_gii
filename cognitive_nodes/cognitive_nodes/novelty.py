import yaml
import numpy as np
from numpy import bool_, float_

from collections import deque
from copy import deepcopy

from cognitive_nodes.drive import Drive
from cognitive_nodes.goal import Goal
from cognitive_nodes.policy import Policy, PolicyBlocking
from core.service_client import ServiceClient, ServiceClientAsync
from core.utils import actuation_dict_to_msg, perception_msg_to_dict

from std_msgs.msg import String
from core_interfaces.srv import GetNodeFromLTM
from cognitive_node_interfaces.srv import Execute, Predict
from cognitive_node_interfaces.msg import Episode as EpisodeMsg


class DriveNovelty(Drive):
    """
    DriveNovelty Class, represents a drive to explore the environment. 
    """    
    def __init__(self, name="drive", class_name="cognitive_nodes.drive.Drive", **params):
        """
        Constructor of the DriveNovelty class.

        :param name: The name of the Drive instance.
        :type name: str
        :param class_name: The name of the Drive class, defaults to "cognitive_nodes.drive.Drive".
        :type class_name: str
        """        
        super().__init__(name, class_name, **params)

    def evaluate(self, perception=None):
        """
        Evaluation that always returns 1.0, as the drive is always .

        :param perception: Unused perception.
        :type perception: dict or Any.
        :return: Evaluation of the Drive.
        :rtype: cognitive_node_interfaces.msg.Evaluation
        """        
        self.evaluation.evaluation = 1.0
        self.evaluation.timestamp = self.get_clock().now().to_msg()
        return self.evaluation
    
class PolicyNovelty(Policy):
    """
    PolicyNovelty Class, represents a policy that selects a random policy from the LTM and executes it.
    """    
    def __init__(self, name='policy_novelty', exclude_list=[], ltm_id=None, **params):
        """
        Constructor of the PolicyNovelty class.

        :param name: Name of the node.
        :type name: str
        :param exclude_list: List of policies that should not be selected for executions, defaults to [].
        :type exclude_list: list
        :param ltm_id: Id of the LTM that includes the nodes.
        :type ltm_id: str
        """        
        super().__init__(name, **params)
        self.policies = PolicyQueue()
        self.LTM_id = ltm_id
        self.exclude_list=exclude_list
        self.exclude_list.append(self.name)
        self.setup()
        self.counter=0
        self.ltm_subscriber = self.create_subscription(String, "/state", self.ltm_change_callback, 1, callback_group=self.cbgroup_client)
        

    def setup(self):
        """
        Setup method that configures the PolicyNovelty node.
        """        
        ltm = self.request_ltm()
        random_seed = getattr(self, 'random_seed', None)
        self.rng = np.random.default_rng(random_seed)
        self.configure_policies(ltm)
        self.get_logger().info("Configured Novelty Policy.")

    def request_ltm(self):
        """
        Requests data from the LTM.

        :return: LTM dump.
        :rtype: dict
        """        
        # Call get_node service from LTM
        service_name = "/" + str(self.LTM_id) + "/get_node"
        request = ""
        client = ServiceClient(GetNodeFromLTM, service_name)
        ltm_response = client.send_request(name=request)
        ltm = yaml.safe_load(ltm_response.data)
        return ltm
    
    def ltm_change_callback(self, msg):
        """
        Callback that reads the LTM change and updates the policies accordingly.

        :param msg: LTM change message.
        :type msg: std_msgs.msg.String
        """        
        self.get_logger().debug("Reading LTM Change...")
        ltm = yaml.safe_load(msg.data)
        self.configure_policies(ltm)

    def configure_policies(self, ltm_cache):
        """
        Creates a list of eligible policies to be executed and shuffles it.

        :param ltm_cache: LTM cache.
        :type ltm_cache: dict
        """        
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
        """
        Selects a policy from the queue. It begins by selecting the front policy and rotates the queue, once all the queue has been iterated, the policies are shuffled.

        :return: Selected policy.
        :rtype: str
        """        
        policy=self.policies.select_policy()
        self.counter+=1

        if self.counter%len(self.policies) == 0:
            self.get_logger().info(f"DEBUG: Shuffling Policies (Counter: {self.counter} Policies: {len(self.policies)})")
            self.policies.shuffle(self.rng)
            self.counter=0

        return policy

    async def execute_callback(self, request, response):
        """
        Callback that selects a policy and then executes it.

        :param request: Execution request
        :type request: cognitive_node_interfaces.srv.Execute.Request
        :param response: Response of the execution. Includes the name of the selected policy.
        :type response: cognitive_node_interfaces.srv.Execute.Response
        :return: Response of the execution.
        :rtype: cognitive_node_interfaces.srv.Execute.Response
        """        
        policy = self.select_policy()
        if policy not in self.node_clients:
            self.node_clients[policy] = ServiceClientAsync(self, Execute, f"policy/{policy}/execute", callback_group=self.cbgroup_client)
        self.get_logger().info('Executing policy: ' + policy + '...')
        await self.node_clients[policy].send_request_async()
        response.policy = policy
        return response
    

class PolicyQueue:
    """
    PolicyQueue Class, wrapper over the builtin deque object to create a policy queue.
    """    
    def __init__(self):
        """
        Constructor of the PolicyQueue class.
        """
        self.queue = deque()

    def select_policy(self):
        """
        Selects a policy from the queue. It begins by selecting the front policy and rotates the queue.

        :return: Selected policy.
        :rtype: str
        """
        policy = self.front()
        self.queue.rotate(1)
        return policy
    
    def shuffle(self, rng: np.random.Generator):
        """
        Shuffles the queue using the provided random number generator.

        :param rng: Random number generator to use for shuffling.
        :type rng: np.random.Generator
        """
        rng.shuffle(self.queue)

    def find_differences(self, items):
        """
        Finds the differences between the current queue and the provided items.

        :param items: List of items to compare with the queue.
        :type items: list
        :return: Tuple containing the new items and the missing items.
        :rtype: tuple (new_items, missing_items)
        """
        new = [x for x in items if x not in self.queue]
        missing = [x for x in self.queue if x not in items]
        return new, missing
    
    def merge(self, items):
        """
        Merges the current queue with the provided items. It adds new items and removes missing items.

        :param items: List of items to merge with the queue.
        :type items: list
        :return: True if there are changes, False otherwise.
        :rtype:  bool
        """
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
        """
        Adds an item to the front of the queue.

        :param item: Item to add to the queue.
        :type item: str
        :return: None
        :rtype: NoneType
        """
        return self.queue.appendleft(item)
    
    def dequeue(self):
        """
        Removes the last item from the queue.

        :return: The last item in the queue.
        :rtype: str
        """
        return self.queue.pop()
    
    def remove(self, item):
        """
        Removes an item from the queue.

        :param item: Item to remove from the queue.
        :type item: str
        :return: True if the item was removed, False otherwise.
        :rtype: bool
        """
        if item in self.queue:
            self.queue.remove(item)
            return True
        return False
    
    def isEmpty(self):
        """
        Checks if the queue is empty.

        :return: True if the queue is empty, False otherwise
        :rtype: bool
        """
        return len(self.queue) == 0
    
    def front(self):
        """
        Returns the first item in the queue.

        :return: The first item in the queue.
        :rtype: str
        """
        return self.queue[-1]
    
    def rear(self):
        """
        Returns the last item in the queue.

        :return: The last item in the queue.
        :rtype: str
        """
        return self.queue[0]
    
    def exists(self, item):
        """
        Checks if an item exists in the queue.

        :param item: Item to check for existence in the queue.
        :type item: str
        :return: True if the item exists in the queue, False otherwise.
        :rtype: bool
        """
        return item in self.queue
    
    def __len__(self):
        """
        Returns the length of the queue.

        :return: The length of the queue.
        :rtype: int
        """
        return len(self.queue)
    

class PolicyRandomAction(PolicyBlocking):
    """
    WORK IN PROGRESS

    PolicyRandomAction Class, represents a policy that executes a random low level action.
    """    
    def __init__(self, name='policy_random_action', actuation_config=None, **params):
        """
        Constructor of the PolicyRandomAction class.

        :param name: Name of the policy node.
        :type name: str
        :param actuation_config: Dictionary with the existing actuators and its data type.
        :type actuation_config: dict
        """        
        super().__init__(name, **params)
        self.actuation_config=actuation_config
        self.actuation={} #All fields are normalized 0 to 1
        self.setup()

    def setup(self):
        """
        Setup method that configures the PolicyRandomAction node.
        :raises TypeError: Unknown type assigned to an actuator.
        """        
        self.world_model_client=ServiceClientAsync(self, Predict, "/world_model/GRIPPER_AND_LOW_FRICTION/predict", self.cbgroup_client) #TODO: Change world model service to a parameter
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
        """
        Randomizes the actuation values.

        :raises TypeError: Unknown type assigned to an actuator.
        """        
        for actuator in self.actuation:
            for param in self.actuation[actuator][0]:
                if self.actuation_config[actuator][param]["type"]=="float":
                    self.actuation[actuator][0][param]=self.rng.uniform()
                elif self.actuation_config[actuator][param]["type"]=="bool":
                    self.actuation[actuator][0][param]=self.rng.choice([True,True,True,True,True,False])
                else:
                    self.get_logger().info(f"DEBUG: {actuator}, {param} {self.actuation[actuator][0][param]} type: {type(self.actuation[actuator][0][param])}")
                    raise TypeError("Actuation parameter is of unknown type")
                self.get_logger().info(f"DEBUG: {actuator}, {param} : {self.actuation[actuator][0][param]}")


    async def execute_callback(self, request, response):
        """
        Makes a service call to the server that handles the execution of the policy.

        :param request: The request to execute the policy.
        :type request: cognitive_node_interfaces.srv.Execute.Request
        :param response: The response indicating the executed policy.
        :type response: cognitive_node_interfaces.srv.Execute.Response
        :return: The response with the executed policy name.
        :rtype: cognitive_node_interfaces.srv.Execute.Response
        """
        self.get_logger().info('Executing policy: ' + self.name + '...')
        self.randomize_actuation()
        actuation_msg=actuation_dict_to_msg(self.actuation)
        input_episode = EpisodeMsg()
        input_episode.old_perception = request.perception
        input_episode.action.actuation = actuation_msg
        result = await self.world_model_client.send_request_async(input_episodes=[input_episode])
        actuation_msg=actuation_dict_to_msg(self.actuation)
        await self.policy_service.send_request_async(action=actuation_msg)
        response.policy=self.name
        response.action=actuation_dict_to_msg(self.actuation)
        return response
