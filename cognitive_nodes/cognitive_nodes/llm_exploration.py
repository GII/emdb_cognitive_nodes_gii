import yaml
import numpy as np
from ollama import chat, Client
import json
import re
import ast

from collections import deque
from copy import deepcopy

from cognitive_nodes.drive import Drive
from cognitive_nodes.goal import Goal
from cognitive_nodes.policy import Policy, PolicyBlocking
from core.service_client import ServiceClient, ServiceClientAsync
from core.utils import actuation_dict_to_msg, perception_msg_to_dict, actuation_msg_to_dict, EncodableDecodableEnum

from std_msgs.msg import String
from core_interfaces.srv import GetNodeFromLTM
from cognitive_node_interfaces.srv import Execute, Predict
from cognitive_node_interfaces.msg import Episode as EpisodeMsg
from cognitive_processes_interfaces.msg import ControlMsg
from simulators.pump_panel_sim_discrete import PumpObjects

class DriveLLMExploration(Drive):
    """
    DriveLLMExploration Class, represents a drive to explore the environment with LLMs.
    """    
    def __init__(self, name="drive", class_name="cognitive_nodes.drive.Drive", **params):
        """
        Constructor of the DriveLLMExploration class.

        :param name: The name of the Drive instance.
        :type name: str
        :param class_name: The name of the Drive class, defaults to "cognitive_nodes.drive.Drive".
        :type class_name: str
        """        
        super().__init__(name, class_name, **params)

    def evaluate(self, perception=None):
        """
        Evaluation that always returns 1.0, as the drive is always.

        :param perception: Unused perception, defaults to None.
        :type perception: dict or NoneType
        :return: Evaluation of the Drive. It always returns 1.0.
        :rtype: cognitive_node_interfaces.msg.Evaluation
        """        
        self.evaluation.evaluation = 1.0
        self.evaluation.timestamp = self.get_clock().now().to_msg()
        
        return self.evaluation
    
    
class PolicyLLMExploration(Policy):
    def __init__(self, name='policy_llm_exploration', model = None, client_host = None, temperature = 0.1, num_predict = 8, initial_prompts = [], exp_stages = None, max_episodes = 5, ltm_id = None, **params):
        """
        Constructor of the PolicyLLMExploration class.

        :param name: Name of the node
        :type name: str
        :param model: The model to be used by the LLM.
        :type model: str
        :param client_host: Host of the LLM client.
        :type client_host: str
        :param temperature: Temperature for the LLM.
        :type temperature: float
        :param num_predict: Number of tokens to answer by the LLM.
        :type num_predict: int
        :param initial_prompts: Initial prompts to be used by the LLM.
        :type initial_prompts: list
        :param exp_stages: Stages of the experiment to change the initial prompt.
        :type exp_stages: list
        :param max_episodes: Maximum number of episodes to be sent to the LLM.
        :type max_episodes: int
        :param ltm_id: Id of the LTM that includes the nodes
        :type ltm_id: str
        """
        super().__init__(name, **params)
        self.LTM_id = ltm_id
        self.initial_prompts = initial_prompts
        self.initial_prompt = initial_prompts[0]
        self.max_episodes = max_episodes
        self.model = model
        self.temperature = temperature
        self.num_predict = num_predict
        self.exp_stages = exp_stages
        self.client = Client(host=client_host) #TODO Add the posibility of using OpenAI API.
        self.episodes_subscriber = self.create_subscription(EpisodeMsg, "/main_loop/episodes", self.episodes_callback, 1, callback_group=self.cbgroup_client)
        self.control_subscriber = self.create_subscription(ControlMsg, "/main_loop/control", self.control_callback, 1, callback_group=self.cbgroup_client)
        self.episodes = []
        self.policies = self.configure_policies()
        random_seed = getattr(self, 'random_seed', None)
        self.rng = np.random.default_rng(random_seed)

    def request_ltm(self):
        """
        Requests data from the LTM.
        """        
        # Call get_node service from LTM
        service_name = "/" + str(self.LTM_id) + "/get_node"
        request = ""
        client = ServiceClient(GetNodeFromLTM, service_name)
        ltm_response = client.send_request(name=request)
        ltm = yaml.safe_load(ltm_response.data)
        return ltm
    
    def configure_policies(self):
        """
        Creates a list of eligible policies to be executed and shuffles it.
        """
        ltm_cache = self.request_ltm()        
        policies = list(ltm_cache["Policy"].keys())
        self.get_logger().info(f"Configuring Policies: {policies}") #TODO: Possibility of using new policies added in LTM
        return policies
    

    def read_reward(self, reward_list):
        """
        Reads the reward from the reward list. The reward is True if any of the keys in the reward list contains "goal" and its value is 1.0.
        A child class could be implemented to read the reward in a different way.

        :param reward_list: Dictionary with the reward list.
        :type reward_list: dict
        :return: True if any of the keys in the reward list contains "goal" and its value is 1.0, otherwise False.
        :rtype: bool
        """
        reward = False
        for key, value in reward_list.items():
            if "goal" in key and value == 1.0:
                reward = True
        return reward

    def control_callback(self, msg:ControlMsg):
        """
        Callback that processes the control message. It updates the initial prompt based on the current iteration and resets the episodes if the command is 'reset_world'.

        :param msg: Message containing the control information.
        :type msg: cognitive_processes_interfaces.msg.ControlMsg
        """
        self.iteration = msg.iteration
        command = msg.command
        if command == 'reset_world':
            self.episodes = []
        
        if self.exp_stages:
            for i in range(len(self.exp_stages)):
                if self.iteration <= self.exp_stages[i]:
                    self.initial_prompt = self.initial_prompts[i]
                    break
                else:
                    self.initial_prompt = self.initial_prompts[-1]
        
    def episodes_callback(self, msg:EpisodeMsg):
        """
        Callback that processes the episode message.
        :param msg: Message containing the episode information.
        :type msg: cognitive_node_interfaces.msg.Episode
        """
        old_perception_msg = msg.old_perception
        old_perception = perception_msg_to_dict(old_perception_msg)
        policy = msg.policy
        perception_msg = msg.perception
        perception = perception_msg_to_dict(perception_msg)
        reward_list = yaml.safe_load(msg.reward_list)
        reward = self.read_reward(reward_list)
        episode = [old_perception, policy, perception, reward]
        self.episodes.append(episode)
        if len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)
    
    def send_to_LLM(self):
        """
        Sends the initial prompt and the episodes to the LLM, and returns the generated text.

        :return: The generated text from the LLM.
        :rtype: str
        """
        #TODO: Add the posibility of using OpenAI API.
        messages = [self.initial_prompt]
        for episode in self.episodes:
            formatted_episode = self.format_episode(episode)
            command = {'role':'user', 'content': formatted_episode}
            messages.append(command)
        self.get_logger().info(f"EPISODES: {messages}")
        response = self.client.chat(model=self.model, messages=messages, options={'temperature': self.temperature, "num_predict":self.num_predict})
        generated_text = response.message.content
        text = generated_text.strip()
        self.get_logger().info(f"LLM RESPONSE: {text}")
        return text
    
    async def execute_callback(self, request, response):
        """
        Callback that processes the request to execute a policy. It sends the perception to the LLM, receives the policy to execute, and executes it.

        :param request: Request to execute a policy, containing the perception.
        :type request: cognitive_node_interfaces.srv.Execute.Request
        :param response: Response that contain the executed policy.
        :type response: cognitive_node_interfaces.srv.Execute.Response
        :return: Response that contains the executed policy.
        :rtype:  cognitive_node_interfaces.srv.Execute.Response
        """
        self.get_logger().info("Sending request to LLM...")
        if not self.episodes:
            self.episodes = [[None, None, perception_msg_to_dict(request.perception), False]]
        policy = self.send_to_LLM() # LLM has to answer with the name of the policy
        self.get_logger().info(f"POLICIES IN LTM: {self.policies}")
        if policy not in self.policies:
            self.get_logger().error("LLM DID NOT RETURN A VALID POLICY. CHOOSING RANDOMLY...")
            policy = self.rng.choice(self.policies)
        
        if policy not in self.node_clients:
            self.node_clients[policy] = ServiceClientAsync(self, Execute, f"policy/{policy}/execute", callback_group=self.cbgroup_client)
        self.get_logger().info('Executing policy: ' + policy + '...')
        await self.node_clients[policy].send_request_async()
        response.policy = policy
        return response
    
    def format_episode(self, episode):
        """
        Formats the episode into a YAML string that can be sent to the LLM.
        This method should be implemented in child classes to format the episode in a specific way.

        :param episode: List containing the old perception, action, new perception, and reward.
        :type episode: list
        :raises NotImplementedError: This method should be implemented in child classes.
        """
        raise NotImplementedError

    
class PolicyLLMExplorationFruitShop(PolicyLLMExploration):
    """
    PolicyLLMExplorationFruitShop Class, represents a policy to explore the Fruit Shop experiment environment with LLMs.
    """

    def read_reward(self, reward_list):
        """
        Reads the reward from the reward list, taking into account the stages of the Fruit Shop experiment.
        :param reward_list: Dictionary with the reward list.
        :type reward_list: dict
        :return: True if there is a reward, False otherwise.
        :rtype: bool
        """
        reward = False

        if self.iteration <= self.exp_stages[0] or (self.iteration > self.exp_stages[1] and self.iteration <= self.exp_stages[2]):
            for key, value in reward_list.items():
                if "effect" in key and value == 1.0:
                    reward = True
        elif (self.iteration > self.exp_stages[0] and self.iteration <= self.exp_stages[1]) or (self.iteration > self.exp_stages[2]):
            for key, value in reward_list.items():
                if "goal" in key and value == 1.0:
                    reward = True

        return reward
    
        
    def format_episode(self, episode):
        """
        Formats the episode into a YAML string that can be sent to the LLM.
        Specific for the Fruit Shop experiment.

        :param episode: List containing the old perception, action, new perception, and reward.
        :type episode: list
        :raises NotImplementedError: This method should be implemented in child classes.
        """
        
        old_perception, action, new_perception, reward = episode
        
        if old_perception and action:
            formatted_episode = {
                "old_state": {
                    "fruit_in_left_hand": old_perception["fruit_in_left_hand"][0]["data"],
                    "fruit_in_right_hand": old_perception["fruit_in_right_hand"][0]["data"],
                    "fruits": old_perception["fruits"][0],
                    "scales": old_perception["scales"][0],
                    "button_light": old_perception["button_light"][0]["data"]
                },
                "action": action,
                "current_state": {
                    "fruit_in_left_hand": new_perception["fruit_in_left_hand"][0]["data"],
                    "fruit_in_right_hand": new_perception["fruit_in_right_hand"][0]["data"],
                    "fruits": new_perception["fruits"][0],
                    "scales": new_perception["scales"][0],
                    "button_light": new_perception["button_light"][0]["data"]
                },
                "goal_reached": reward
            }
        else:
            formatted_episode = {
                "old_state": None,
                "action": None,
                "current_state": {
                    "fruit_in_left_hand": new_perception["fruit_in_left_hand"][0]["data"],
                    "fruit_in_right_hand": new_perception["fruit_in_right_hand"][0]["data"],
                    "fruits": new_perception["fruits"][0],
                    "scales": new_perception["scales"][0],
                    "button_light": new_perception["button_light"][0]["data"]
                },
                "goal_reached": reward
            }

        return yaml.dump(formatted_episode, default_flow_style=False, sort_keys=False)
    