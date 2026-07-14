from xmlrpc import client
import yaml
import numpy as np
import re

from math import isclose
from copy import copy
from rclpy.node import Node
from cognitive_nodes.drive import Drive
from cognitive_nodes.goal import Goal, GoalMotiven, GoalLearnedSpace
from cognitive_nodes.policy import Policy
from core.service_client import ServiceClient, ServiceClientAsync

from std_msgs.msg import String
from core_interfaces.srv import GetNodeFromLTM, CreateNode, UpdateNeighbor
from cognitive_node_interfaces.msg import SuccessRate, PerceptionStamped
from cognitive_node_interfaces.srv import GetActivation, SendSpace, GetEffects, ContainsSpace, SetActivation
from core.utils import perception_dict_to_msg, perception_msg_to_dict, compare_perceptions
from cognitive_nodes.utils import PNodeSuccess, EpisodeSubscription

from ament_index_python.packages import get_package_share_directory

import tkinter as tk
from tkinter import scrolledtext,messagebox
import os
from ollama import chat, Client
from openai import OpenAI
import yamlloader



class DriveAlignment(Drive):
    
    """
    DriveAlignment Class, represents a drive to align the human porpuse to the robot's purpose (missions and drives).
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
        Evaluation that always returns 1.0, as the drive is always active.

        :param perception: Unused perception, defaults to None.
        :type perception: dict or NoneType
        :return: Evaluation of the Drive. It always returns 1.0.
        :rtype: cognitive_node_interfaces.msg.Evaluation
        """        
        self.evaluation.evaluation = 1.0
        self.evaluation.timestamp = self.get_clock().now().to_msg()
        return self.evaluation
    

class PolicyAlignment(Policy):
    """
    Policy that executes the alignment interaction with the user and generates and initializes the missions and drives in the cognitive architecture. 

    This class inherits from the general Policy class.
    """  
    def __init__(self, name='policy', class_name='cognitive_nodes.policy.Policy', prompts_file=None, **params):
        """
        Constructor of the PolicyAlignment class.

        :param name: Name of the node.
        :type name: str
        :param class_name: The name of the base Policy class, defaults to 'cognitive_nodes.policy.Policy'.
        :type class_name: str
        :param prompts_file: YAML file containing the different prompts for the alignment process.
        :type prompts_file: str
        """        
        super().__init__(name, class_name, **params)
        self.config_dir = os.path.join(get_package_share_directory('cognitive_nodes'), 'config')
        self.missions = None
        self.drives = None
        self.sensors = None
        self.perceptions = None
        self.policy_executed = False

        self.chosen_conversation_file = None
        self.conversation = []
        self.final_purpose = None
        self.llm_mission = None
        self.LLM4drives = None
        self.drive_conversation = []
        self.chosen_conversation_file = None
        if not prompts_file:
            raise ValueError("You must provide a prompts_file parameter.")

        self.prompts_file = self.find_folder(prompts_file)
        self.alignment_prompt, self.drives_prompt, self.missions_prompt, self.needs, self.perceptions = self._load_prompts()

    def find_folder(self, filename):
        """
        Finds the config folder direction on the computer.

        :param filename: Name of the file to find.
        :type filename: str
        :return: Full path to the file.
        :rtype: str
        """ 
        file_path = os.path.join(self.config_dir, filename)
        if not os.path.exists(file_path):
            self.get_logger().fatal(f"{self.config_dir}")
            raise FileNotFoundError(f"The file {filename} was not found in the config directory.")
        return file_path

    def _load_prompts(self):
        """
        Extract the different prompts from the YAML file.

        :return: alignment_prompt, drives_prompt, missions_prompt, needs, perceptions
        :rtype: tuple[str, str, str, str, str]
        """
        with open(self.prompts_file, "r", encoding="utf-8") as file:
            prompts = yaml.load(file, Loader=yaml.FullLoader)
            alignment_prompt = prompts.get('alignment_prompt')
            drives_prompt = prompts.get('drives_prompt')
            missions_prompt = prompts.get('missions_prompt')
            needs = prompts.get('internal_needs')
            perceptions = prompts.get('perceptions')
        return alignment_prompt, drives_prompt, missions_prompt, needs, perceptions

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
        final_answer, self.missions, self.drives = self.interface()
        self.get_logger().info(f'Missions: {self.missions}')
        self.get_logger().info(f'Drives: {self.drives}')
        await self.create_missions_drives(self.missions, self.drives)
        response.policy = self.name
        if self.policy_executed:
            self.get_logger().info('Policy executed successfully.')
            client = ServiceClientAsync(self, SetActivation, '/robot_purpose/alignment_need/set_activation', self.cbgroup_client)
            await client.send_request_async(activation=0.0)
        return response
    
    async def create_missions_drives(self, missions, drives):
        """
        Takes the missions and drives generated by the LLM and creates the corresponding Robot Purpose and Drive nodes in the cognitive architecture.

        :param missions: List of missions generated by the LLM.
        :type missions: list
        :param drives: List of drives generated by the LLM.
        :type drives: list
        """
        self.get_logger().info('Creating missions and drives...')
        for index, (tag, weight) in enumerate(missions):
            mission_name = f"{tag}_mission"
            drive_id = f"{tag}_drive"
            purpose_type = 'Mission'
            if index == len(missions) - 1:
                terminal = True
            else:
                terminal = False
            
            mission_parameters = {
                "weight": weight,
                "drive_id": drive_id,
                "purpose_type": purpose_type,
                "terminal": terminal
            }
            
            drive_parameters = {
                "drive_function": drives[index],
                "neighbors": [{"name": mission_name, "node_type": "RobotPurpose"}]
            }

            mission = await self.create_node_client(name=mission_name, class_name="cognitive_nodes.robot_purpose.AlignmentMission", parameters=mission_parameters)
            drive = await self.create_node_client(name=drive_id, class_name="cognitive_nodes.drive.DriveLLM", parameters=drive_parameters)

            if mission and drive:
                self.get_logger().info(f"Created need and drive: {tag}")
                self.get_logger().info(f"Mission: {mission_name}, Drive: {drive_id}")
                self.policy_executed = True
            if not (mission and drive):
                self.get_logger().fatal(f"Failed creation of the operational need and drive: {tag}")
                self.policy_executed = False
        return 
    
    def interface(self):
        """
        Starts the interface that interacts with the user to align the human's purpose with the robot's purpose.

        :return: final_answer, missions, drives
        :rtype: tuple[str, list, list]
        """
        path = init_conversation_file()
        model, host, initial_prompt = load_configuration(self.alignment_prompt)
        LLM = LLMmodel(model, host, initial_prompt)
        root = tk.Tk()
        results = ChatInterface(root, LLM, self.perceptions, self.needs, self.alignment_prompt, self.missions_prompt, self.drives_prompt, path)
        root.mainloop()
        missions, drives = extract_missions_and_drives(results.drives)
        return results.drives, missions, drives    


class LLMmodel():
    """
    LLMmodel class creates the LLM model instance and handles interactions.
    """  
    """
    To correctly execute this code, you need to set the environment variable OPENAI_API_KEY with your OpenAI API key.
    Execute the following commands in your terminal:
        echo "export OPENAI_API_KEY='insert_your_api_key_here'" >> ~/.bashrc
        source ~/.bashrc
    """
    def __init__(self, model, host, initial_prompt):
        """Constructor of the LLMmodel class.

        :param model: Model used for the LLM.
        :type model: str
        :param host: API host URL
        :type host: str
        :param initial_prompt: Prompt to initialize the LLM.
        :type initial_prompt: str
        """        
        self.model = model
        self.host = host
        self.initial_prompt = initial_prompt
        if model == "phi4:14b" or model == "Qwen3:30b":
            self.client = Client(host=self.host)
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(base_url=self.host,
                                 api_key=api_key,)
    @staticmethod
    def LLM_worker(client:Client, model, command):
        """
        Extracts the response from the LLM based on the model used.

        :param client: LLM client instance.
        :type client: Client
        :param model: Model used for the LLM.
        :type model: str
        :param command: Command or messages to send to the LLM.
        :type command: list
        :return: Generated text from the LLM.
        :rtype: str
        """
        if model == "phi4:14b" or model == "Qwen3:30b":
            response = client.chat(model=model, messages=command, options={'temperature': 0.1})
            return response.message.content
        else:
            response = client.chat.completions.create(model="openai/gpt-4.1:floor", messages=command)
            return response.choices[0].message.content
    
    def send_to_LLM(self, conversation, extra_context):
        """
        Sends the conversation and extra context to the LLM and returns the generated text.

        :param conversation: Conversation history.
        :type conversation: list
        :param extra_context: Additional context to provide to the LLM.
        :type extra_context: list
        :return: Generated text from the LLM.
        :rtype: str
        """
        messages = [self.initial_prompt]
        messages.extend(extra_context)
        messages.extend(conversation)  
        generated_text = self.LLM_worker(self.client, self.model, messages)
        return generated_text


class ChatInterface():
    """
    ChatInterface class creates a GUI with the different models for user interaction.
    """
    def __init__(self, root, LLM, perceptions, needs, alignment_prompt, missions_prompt, drives_prompt, path):
        """
        Constructor of the ChatInterface class.

        :param root: Root window of the GUI.
        :type root: tk.Tk
        :param LLM: LLM model instance.
        :type LLM: LLMmodel
        :param perceptions: prompt with the perceptions to provide to the LLM.
        :type perceptions: str
        :param needs: prompt with the internal needs to provide to the LLM.
        :type needs: str
        :param alignment_prompt: prompt for the alignment LLM.
        :type alignment_prompt: str
        :param missions_prompt: prompt for the missions generation LLM.
        :type missions_prompt: str
        :param drives_prompt: prompt for the drives generation LLM.
        :type drives_prompt: str
        :param path: Path to save the conversation.
        :type path: str
        """
        self.root = root
        self.root.title("Chat Interface")
        self.LLM = LLM
        self.conversation = []
        self.final_purpose = None
        self.llm_mission = None
        self.LLM4drives = None
        self.drives = None
        self.awaiting_drive_feedback = False
        self.drive_conversation = []

        self.perceptions = perceptions
        self.needs = needs
        self.alignment_prompt = alignment_prompt
        self.missions_prompt = missions_prompt
        self.drives_prompt = drives_prompt
        self.path = path

        # Text area for conversation history
        self.chat_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled', width=90, height=30, bg="#f4f4f4", font=("Arial", 11))
        self.chat_display.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

        # Multiline input field
        self.user_input = tk.Text(root, width=70, height=4, font=("Arial", 12))
        self.user_input.grid(row=1, column=0, padx=10, pady=10)

        # Send button
        self.user_input.bind("<Control-Return>", self.send_message)
        self.send_button = tk.Button(root, text="Send", command=self.send_message, width=10, bg="#48C9F1", fg="white")
        self.send_button.grid(row=1, column=1, padx=10, pady=10)
        self.conversation = []
    
    def display_message(self, role, message):
        """
        Displays a message in the chat display area.

        :param role: Role of the message sender ('user' or 'assistant').
        :type role: str
        :param message: Message content to display.
        :type message: str

        """
        self.chat_display.configure(state='normal')
        self.chat_display.insert(tk.END, f"{role.capitalize()}: {message}\n\n")
        self.chat_display.configure(state='disabled')
        self.chat_display.yview(tk.END)

    def send_message(self, event=None):
        """
        Recieves the user input, to send it to the alignment LLM, and displays the response. Calls the generate missions and drives methods.

        :param event: Event that triggered the method, defaults to None.
        :type event: tk.Event or NoneType

        """
        user_msg = self.user_input.get("1.0", tk.END).strip()
        if not user_msg:
            return

        self.conversation.append({'role': 'user', 'content': user_msg})
        self.display_message('user', user_msg)
        self.user_input.delete("1.0", tk.END)

        try:
            objects = [{"role": "system", "content": str(self.perceptions["objects"])}]
            reply = self.LLM.send_to_LLM(self.conversation, objects)
            self.conversation.append({'role': 'assistant', 'content': reply})
            self.display_message('assistant', reply)
            save_conversation(self.conversation, self.path)

            if "Final description" in reply:
                self.final_purpose = reply
                messagebox.showinfo("Info", "Final description received. Generating missions...")
                self.generate_missions(objects)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def generate_missions(self, objects):
        """
        Sends the final purpose and internal needs to the LLM model and generates the missions.

        :param objects: Description of objects to provide to the LLM.
        :type objects: str

        """
        needs_list = [{"role": "system", "content": str(self.needs["needs"])}]
        combined_data = [objects, needs_list]
        model, host, missions_p = load_configuration(self.missions_prompt)
        LLM4missions = LLMmodel(model, host, missions_p)

        try:
            mission_response = LLM4missions.send_to_LLM([{'role': 'user', 'content': self.final_purpose}], combined_data)
            self.llm_mission = mission_response
            self.conversation.append({'role': 'assistant', 'content': mission_response})
            self.display_message('assistant', mission_response)
            save_conversation(self.conversation, self.path)

            messagebox.showinfo("Info", "Missions generated. Generating drives...")
            self.generate_drives(objects)
        except Exception as e:
            messagebox.showerror("Mission Generation Error", str(e))

    def generate_drives(self, objects):
        """
        Sends the final purpose and the list of missions to the LLM model and generates the drives.

        :param objects: Description of objects to provide to the LLM.
        :type objects: str

        """
        model, host, drives_p = load_configuration(self.drives_prompt)
        LLM4drives = LLMmodel(model, host, drives_p)
        self.LLM4drives = LLM4drives
        first_prompt = self.final_purpose + "\n\n" + self.llm_mission
        self.drive_conversation.append({'role': 'user', 'content': first_prompt})
        try:
            drive_response = LLM4drives.send_to_LLM(self.drive_conversation, objects)
            self.conversation.append({'role': 'assistant', 'content': drive_response})
            self.drive_conversation.append({'role': 'assistant', 'content': drive_response})
            self.display_message('assistant', drive_response)
            save_conversation(self.conversation, self.path)

            if "Final drives" in drive_response:
                    self.drives = drive_response
                    messagebox.showinfo("Drives", "Final drives received. Interaction complete.")
                    self.root.after(500, self.root.destroy)
            else:
                self.awaiting_drive_feedback = True
                self.send_button.configure(command=lambda: self.send_drive_feedback(objects))
                self.user_input.bind('<Control-Return>', lambda event: self.send_drive_feedback(objects))
                
        except Exception as e:
            messagebox.showerror("Drives Generation Error", str(e))

    def send_drive_feedback(self, objects):
        """
        Recieves the feedback from the user to refine the drives if needed and finishes the interaction when the final drives are accepted.

        :param objects: Description of objects to provide to the LLM.
        :type objects: str

        """
        if not self.awaiting_drive_feedback:
            return
        feedback = self.user_input.get("1.0", tk.END).strip()
        if not feedback:
            return

        self.user_input.delete("1.0", tk.END)
        self.display_message('user', feedback)
        self.drive_conversation.append({'role': 'user', 'content': feedback})
        self.conversation.append({'role': 'user', 'content': feedback})

        try:
            drive_response = self.LLM4drives.send_to_LLM(self.drive_conversation, objects)
            self.drive_conversation.append({'role': 'assistant', 'content': drive_response})
            self.conversation.append({'role': 'assistant', 'content': drive_response})
            self.display_message('assistant', drive_response)
            save_conversation(self.conversation, self.path)

            if "Final answer" in drive_response:
                self.drives = drive_response
                self.awaiting_drive_feedback = False
                messagebox.showinfo("Drives", "Final drives received. Interaction complete.")
                self.root.after(10000, self.root.destroy)
            else:
                # Stay in feedback loop
                self.awaiting_drive_feedback = True
                self.send_button.configure(command=lambda: self.send_drive_feedback(objects))
                self.user_input.bind('<Control-Return>', lambda event: self.send_drive_feedback(objects))

        except Exception as e:
            messagebox.showerror("Feedback Error", str(e))
            
def init_conversation_file():
    """
    It finds the first available filename and saves it in a global variable.

    :return: Path to the conversation file.
    :rtype: str
    """
    base_name = "conversation_"
    extension = ".yaml"
    dir = os.getcwd()
    i = 1
    while True:
        file_name = f"{base_name}{i}{extension}"
        file_path = os.path.join(dir, file_name)
        if not os.path.exists(file_path):
            return file_path
        i += 1

def save_conversation(conversation, file_path):
    """
    Writes to the same file chosen at init.

    :param conversation: Conversation to save.
    :type conversation: list
    :param file_path: Path to the conversation file.
    :type file_path: str

    """
    if file_path is None:
        raise RuntimeError("You must call init_conversation_file() before save_conversation()")

    with open(file_path, "w", encoding="utf-8") as file:
        yaml.dump(conversation, file, allow_unicode=True)

def load_configuration(config_prompt):
    """
    Extracts the model, host, and initial prompt from the configuration dictionary.

    :param config_prompt: Configuration dictionary.
    :type config_prompt: dict
    :return: model, host, initial_prompt
    :rtype: tuple[str, str, str]
    """
    model = config_prompt["model"]
    host = config_prompt["host"]
    initial_prompt = config_prompt["initial_prompt"]
    return model, host, initial_prompt

def extract_missions_and_drives(text):
    """
    Extracts missions and drives from the LLM response text.

    :param text: LLM response text.
    :type text: str or list
    :return: Extracted missions and drives.
    :rtype: tuple[list, list]
    """    
    if isinstance(text, list):
        text = "\n".join(text)
    missions = []
    drives = []
    mission_blocks = re.findall(
        r'(Mission\d+:\s*\[.*?\]\s*Drive:\s*.*?)(?=(?:\n\s*\n|$))',
        text, re.DOTALL
    )

    for block in mission_blocks:
        tag_match = re.search(r'Mission\d+:\s*\[([^\],]+),\s*([0-9.]+)\]', block)
        drive_match = re.search(r'Drive:\s*(.*)', block)
        if tag_match and drive_match:
            mission_tag = tag_match.group(1)
            mission_value = float(tag_match.group(2))
            drive = drive_match.group(1).strip()

            missions.append([mission_tag, mission_value])
            drives.append(drive)

    return missions, drives
    