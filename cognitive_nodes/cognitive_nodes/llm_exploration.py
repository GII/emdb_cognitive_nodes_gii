import yaml
import numpy as np
from ollama import chat, Client
import json
import re
import ast

from collections import deque
from copy import deepcopy

from cognitive_nodes.need import Need
from cognitive_nodes.drive import Drive
from cognitive_nodes.goal import Goal
from cognitive_nodes.policy import Policy, PolicyBlocking
from core.service_client import ServiceClient, ServiceClientAsync
from core.utils import actuation_dict_to_msg, perception_msg_to_dict, actuation_msg_to_dict, EncodableDecodableEnum

from std_msgs.msg import String
from core_interfaces.srv import GetNodeFromLTM
from cognitive_node_interfaces.srv import Execute, Predict
from cognitive_processes_interfaces.msg import Episode as EpisodeMsg
from cognitive_processes_interfaces.msg import ControlMsg
from simulators.pump_panel_sim_discrete import PumpObjects

class DriveLLMExploration(Drive):
    """
    DriveLLMExploration Class, represents a drive to explore the environment with LLMs
    """    
    def __init__(self, name="drive", class_name="cognitive_nodes.drive.Drive", **params):
        """
        Constructor of the DriveLLMExploration class.

        :param name: The name of the Drive instance
        :type name: str
        :param class_name: The name of the Drive class, defaults to "cognitive_nodes.drive.Drive"
        :type class_name: str
        """        
        super().__init__(name, class_name, **params)

    def evaluate(self, perception=None):
        """
        Evaluation that always returns 1.0, as the drive is always .

        :param perception: Unused perception, defaults to None
        :type perception: dict/NoneType, optional
        :return: Evaluation of the Drive
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

        :return: LTM dump
        :rtype: dict
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

        :param ltm_cache: LTM cache
        :type ltm_cache: dict
        """
        ltm_cache = self.request_ltm()        
        policies = list(ltm_cache["Policy"].keys())
        self.get_logger().info(f"Configuring Policies: {policies}") #TODO: Possibility of using new policies added in LTM
        return policies
    

    def read_reward(self, reward_list):
        reward = False
        for key, value in reward_list.items():
            if "goal" in key and value == 1.0:
                reward = True
        return reward

    def control_callback(self, msg:ControlMsg):
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
        raise NotImplementedError

    
class PolicyLLMExplorationFruitShop(PolicyLLMExploration):
    """
    PolicyLLMExploration Class, represents a policy that selects a policy guided by a LLM and executes it.
    """

    def read_reward(self, reward_list):
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
    
class PolicyLLMExplorationPump(PolicyLLMExploration):
    def __init__(self, name='policy_llm_exploration', model = None, client_host = None, temperature = 0.1, num_predict = 8, initial_prompts = [], exp_stages = None, max_episodes = 5, ltm_id = None, **params):
        super().__init__(name, model, client_host, temperature, num_predict, initial_prompts, exp_stages, max_episodes, ltm_id, **params)
        self.objects = list(PumpObjects.__members__.keys())

    def episodes_callback(self, msg:EpisodeMsg):
        old_perception_msg = msg.old_perception
        old_perception = perception_msg_to_dict(old_perception_msg)
        policy = msg.policy
        param_msg = msg.actuation
        param_dict = actuation_msg_to_dict(param_msg)
        param_coded = param_dict.get('policy_params', [{"object":None}])[0]['object']
        if param_coded is not None:
            parameter = PumpObjects.decode(int(param_coded), normalized=False)
        else:
            parameter = None
        perception_msg = msg.perception
        perception = perception_msg_to_dict(perception_msg)
        reward_list = yaml.safe_load(msg.reward_list)
        reward = self.read_reward(reward_list)
        episode = [old_perception, policy, parameter, perception, reward]
        self.episodes.append(episode)
        if len(self.episodes) > self.max_episodes:
            self.episodes.pop(0)

    def format_episode(self, episode):
        old_perception, action, param, new_perception, reward = episode
        if old_perception and action:
            formatted_episode = {
                "old_state": {
                    "discharge_light": old_perception["panel_objects"][0]["discharge_light"],
                    "emergency_button": old_perception["panel_objects"][0]["emergency_button"],
                    "mode_selector": old_perception["panel_objects"][0]["mode_selector"], 
                    "off_button": old_perception["panel_objects"][0]["off_button"],
                    "on_button": old_perception["panel_objects"][0]["on_button"],
                    "output_flow_dial": old_perception["panel_objects"][0]["output_flow_dial"],
                    "start_button": old_perception["panel_objects"][0]["start_button"],
                    "system_backup_light": old_perception["panel_objects"][0]["system_backup_light"],
                    "test_light": old_perception["panel_objects"][0]["test_light"],
                    "tool_1": old_perception["panel_objects"][0]["tool_1"],
                    "tool_2": old_perception["panel_objects"][0]["tool_2"],
                    "v1_button": old_perception["panel_objects"][0]["v1_button"],
                    "v2_button": old_perception["panel_objects"][0]["v2_button"],
                    "v3_button": old_perception["panel_objects"][0]["v3_button"],
                    "voltage_dial": old_perception["panel_objects"][0]["voltage_dial"]
                },
                "action": action,
                "parameter": param,
                "current_state": {
                    "discharge_light": new_perception["panel_objects"][0]["discharge_light"],
                    "emergency_button": new_perception["panel_objects"][0]["emergency_button"],
                    "mode_selector": new_perception["panel_objects"][0]["mode_selector"], 
                    "off_button": new_perception["panel_objects"][0]["off_button"],
                    "on_button": new_perception["panel_objects"][0]["on_button"],
                    "output_flow_dial": new_perception["panel_objects"][0]["output_flow_dial"],
                    "start_button": new_perception["panel_objects"][0]["start_button"],
                    "system_backup_light": new_perception["panel_objects"][0]["system_backup_light"],
                    "test_light": new_perception["panel_objects"][0]["test_light"],
                    "tool_1": new_perception["panel_objects"][0]["tool_1"],
                    "tool_2": new_perception["panel_objects"][0]["tool_2"],
                    "v1_button": new_perception["panel_objects"][0]["v1_button"],
                    "v2_button": new_perception["panel_objects"][0]["v2_button"],
                    "v3_button": new_perception["panel_objects"][0]["v3_button"],
                    "voltage_dial": new_perception["panel_objects"][0]["voltage_dial"]
                },
                "goal_reached": reward
            }
        else:
            formatted_episode = {
                "old_state": None,
                "action": None,
                "parameter": None,
                "current_state": {
                    "discharge_light": new_perception["panel_objects"][0]["discharge_light"],
                    "emergency_button": new_perception["panel_objects"][0]["emergency_button"],
                    "mode_selector": new_perception["panel_objects"][0]["mode_selector"], 
                    "off_button": new_perception["panel_objects"][0]["off_button"],
                    "on_button": new_perception["panel_objects"][0]["on_button"],
                    "output_flow_dial": new_perception["panel_objects"][0]["output_flow_dial"],
                    "start_button": new_perception["panel_objects"][0]["start_button"],
                    "system_backup_light": new_perception["panel_objects"][0]["system_backup_light"],
                    "test_light": new_perception["panel_objects"][0]["test_light"],
                    "tool_1": new_perception["panel_objects"][0]["tool_1"],
                    "tool_2": new_perception["panel_objects"][0]["tool_2"],
                    "v1_button": new_perception["panel_objects"][0]["v1_button"],
                    "v2_button": new_perception["panel_objects"][0]["v2_button"],
                    "v3_button": new_perception["panel_objects"][0]["v3_button"],
                    "voltage_dial": new_perception["panel_objects"][0]["voltage_dial"]
                },
                "goal_reached": reward
        }

        return yaml.dump(formatted_episode, default_flow_style=False, sort_keys=False)


    
    def process_response(self, response):
        match = re.search(r'{.*}', response, re.DOTALL)
        if match:
            policy_str = match.group(0)
        else:
            return None, None

        try:
            policy_dict = ast.literal_eval(policy_str)

            policy = policy_dict["action"]
            parameter = policy_dict["parameter"]

            if parameter is not None:
                parameter = str(parameter).upper()

            return policy, parameter

        except Exception as e:
            return None, None


    async def execute_callback(self, request, response:Execute.Response):
        self.get_logger().info("Sending request to LLM...")
        if not self.episodes:
            self.episodes = [[None, None, None, perception_msg_to_dict(request.perception), False]]
        LLM_response = self.send_to_LLM()
        policy, parameter = self.process_response(LLM_response)
        self.get_logger().info(f"LLM SELECTED POLICY: {policy}({parameter})")
        if policy not in self.policies:
            self.get_logger().error("LLM DID NOT RETURN A VALID POLICY. CHOOSING RANDOMLY...")
            policy = self.rng.choice(self.policies)
            parameter = self.rng.choice(self.objects)
        if policy not in self.node_clients:
            self.node_clients[policy] = ServiceClientAsync(self, Execute, f"policy/{policy}/execute", callback_group=self.cbgroup_client)
        parameter_coded = PumpObjects.encode(parameter, normalized=False)
        param_dict = {'policy_params': [{"object":parameter_coded}]}
        parameter_action = actuation_dict_to_msg(param_dict)
        self.get_logger().info('Executing policy: ' + policy + '...')
        await self.node_clients[policy].send_request_async(parameter = parameter_action)
        response.policy = policy
        response.action = parameter_action
        return response
        
    


    