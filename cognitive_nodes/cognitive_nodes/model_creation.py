import threading
import numpy as np
from math import isclose
from copy import deepcopy
from rclpy.time import Time

from core.service_client import ServiceClientAsync
from core.utils import perception_dict_to_msg
from cognitive_nodes.episode import Episode, Action, episode_msg_to_obj, episode_obj_to_msg, episode_obj_list_to_msg_list
from cognitive_nodes.episodic_buffer import EpisodicBuffer
from cognitive_nodes.drive import Drive
from cognitive_nodes.policy import Policy
from cognitive_nodes.utils import LTMSubscription, EpisodeSubscription

from cognitive_node_interfaces.srv import AddPoints, AddTrace


class ModelCreationMixin(LTMSubscription, EpisodeSubscription):
    def configure_model_creation(self, episode_topic, episode_msg):
        self.missing_world_model = False
        self.LTM_cache = {}
        self.unlinked_drives = []
        self.configure_ltm_subscription(self.LTM_id, self.cbgroup_server)
        self.configure_episode_subscription(episode_topic=episode_topic, episode_msg=episode_msg, callback_group=self.cbgroup_server)
        self.default_class = {}
        self.default_params = {}
        self.setup_connectors()

    def read_ltm(self, ltm_dump):
        """
        Reads the Long-Term Memory (LTM) and populates the LTM cache.

        :param ltm_dump: LTM dump to be used.
        :type ltm_dump: dict
        """
        self.get_logger().info("Reading nodes from LTM: " + self.LTM_id + "...")
        #Add missing elements from LTM to LTM Cache
        for node_type in ltm_dump.keys():
            if self.LTM_cache.get(node_type, None) is None:
                self.LTM_cache[node_type] = {}
            for node in ltm_dump[node_type].keys():
                if self.LTM_cache[node_type].get(node, None) is None:
                    self.LTM_cache[node_type][node] = dict(activation = 0.0, activation_timestamp = 0, neighbors = ltm_dump[node_type][node]["neighbors"])
                    if node_type == "WorldModel":
                        self.create_activation_input({"name": node, "node_type": node_type})
                else: #If node exists update data (except activations)
                    node_data = ltm_dump[node_type][node]
                    del node_data["activation"]
                    del node_data["activation_timestamp"]
                    self.LTM_cache[node_type][node].update(node_data) 
        
        #Remove elements in LTM Cache that were removed from LTM.
        for node_type in self.LTM_cache.keys():
            for node in self.LTM_cache[node_type]:
                if ltm_dump[node_type].get(node, None) is None:
                    del self.LTM_cache[node_type][node]
                    self.delete_activation_input({node: node_type})

        # Check if there are any drives not linked to goals
        self.unlinked_drives = self.get_unlinked_drives()

        world_models = self.LTM_cache.get("WorldModel", None)
        if world_models is None or not world_models:
            self.get_logger().info("No World Model found in LTM.")
            self.missing_world_model = True

    def get_unlinked_drives(self):
            """
            This method retrieves the drives that are not linked to any goal in the LTM cache.

            :return: List of unlinked drives. If there are no unlinked drives, it returns an empty list.
            :rtype: list
            """
            drives=self.LTM_cache.get("Drive", None)
            goals=self.LTM_cache.get("Goal", None)
            if drives:
                drives_list=list(drives.keys())
                for goal in goals:
                    neighbors=goals[goal]["neighbors"]
                    for neighbor in neighbors:
                        if neighbor["name"] in drives_list:
                            drives_list.remove(neighbor["name"])
                return drives_list
            else:
                return []
            
    def linked_cnode(self, goal):
        if goal is None:
            return False
        for node in self.LTM_cache.get("CNode", {}).values():
            neighbors = [neighbor["name"] for neighbor in node["neighbors"] if neighbor["node_type"] == "Goal"]
            if goal in neighbors:
                return True
        return False
    
    def setup_connectors(self):
        """
        Configures the default classes for the cognitive nodes.
        """
        if hasattr(self, "Connectors"):
            for connector in self.Connectors:
                self.default_class[connector["data"]] = connector.get("default_class")
                self.default_params[connector["data"]] = connector.get("parameters", {})

    def generate_node_name(self, node_type):
        index = 0
        while True:
            name = f"{node_type}_{index}"
            if name not in self.LTM_cache.get(node_type, {}):
                return name
            index += 1

    def get_max_activation_node(self, node_type):
        nodes = [{node: self.activation_inputs[node]["data"].activation} for node in self.activation_inputs if self.activation_inputs[node]["node_type"] == node_type]
        if not nodes:
            return None
        max_node = max(nodes, key=lambda x: list(x.values())[0])
        return list(max_node.keys())[0]


class ModelCreationDrive(Drive, ModelCreationMixin):
    def __init__(self, name="model_creation_drive", class_name="cognitive_nodes.drive.Drive", max_iterations=20, episodes_topic=None, episodes_msg=None, model_creation_policy=None, **params):
        super().__init__(name, class_name, **params)
        #self.episodic_buffer = EpisodicBuffer(self, inputs=["old_perception", "action", "perception", "reward_list"], main_size=max_iterations)
        if model_creation_policy is None:
            raise ValueError("Model Creation Policy must be provided.")
        else:
            self.model_creation_policy = model_creation_policy
        self.missing_goal = False
        self.missing_utility_model = False
        self.configure_model_creation(episodes_topic, episodes_msg)


    def episode_callback(self, msg):
        episode = episode_msg_to_obj(msg)
        self.get_logger().debug(f"Received episode with parent policy: {episode.parent_policy} and rewards: {episode.reward_list}")
        if not episode.parent_policy: # Parent policy is empty if no specific Utility Model/Policy is being executed
            for goal, reward in episode.reward_list.items():
                if not isclose(reward, 0.0):
                    if goal in self.unlinked_drives:
                        self.get_logger().info(f"Unlinked drive found: {goal}. Goal node to be created.")
                        self.missing_goal = True
                    elif not self.linked_cnode(goal): # TODO: Also consider the activation of the C-Node's WorldModel
                        self.get_logger().info(f"Goal {goal} not linked to any CNode. Utility Model to be created.")
                        self.missing_utility_model = True
        elif episode.parent_policy == self.model_creation_policy:
            self.get_logger().debug("Model Creation Policy executed. Resetting drive")
            self.missing_world_model = False
            self.missing_goal = False
            self.missing_utility_model = False
    
    def evaluate(self, perception=None):
        if self.missing_goal or self.missing_utility_model or self.missing_world_model:
            self.evaluation.evaluation = 1.0
        else:
            self.evaluation.evaluation = 0.0
        self.evaluation.timestamp = self.get_clock().now().to_msg()
        return self.evaluation
    
    def calculate_activation(self, perception=None, activation_list=None):
        """
        Returns the the activation value of the Drive.

        :param perception: The given perception.
        :type perception: dict
        :return: The activation of the instance and its timestamp.
        :rtype: cognitive_node_interfaces.msg.Activation
        """
        filtered_activation_list = {node:activation for node, activation in activation_list.items() if activation["node_type"] != "WorldModel"}
        self.calculate_activation_max(filtered_activation_list)
        self.evaluate()
        self.activation.activation=self.activation.activation*self.evaluation.evaluation
        timestamp_activation = Time.from_msg(self.activation.timestamp).nanoseconds
        timestamp_evaluation = Time.from_msg(self.activation.timestamp).nanoseconds
        if timestamp_evaluation<timestamp_activation:
            self.activation.timestamp = self.evaluation.timestamp
        return self.activation     
    
    def read_activation_callback(self, msg):
        super().read_activation_callback(msg)
        # Process world model activations
        updated_activations = all((self.activation_inputs[node_name]['updated'] for node_name in self.activation_inputs)) 
        self.get_logger().debug(f"Updated activations: {updated_activations}")
        if updated_activations:
            world_model_activations = [not isclose(activation["data"].activation, 0.0) for activation in self.activation_inputs.values() if activation["node_type"] == "WorldModel"]
            if not any(world_model_activations):
                self.get_logger().debug("No active World Model found. New World Model to be created.")
                self.missing_world_model = True
            else:
                self.get_logger().debug("Active World Model found.")
                self.missing_world_model = False


class ModelCreationPolicy(Policy, ModelCreationMixin):
    def __init__(self, name="model_creation", class_name="cognitive_nodes.drive.Policy", max_iterations=20, episodes_topic=None, episodes_msg=None, **params):
        super().__init__(name, class_name, **params)
        self.episodic_buffer = EpisodicBuffer(self, inputs=["old_perception", "action", "perception"], main_size=max_iterations, secondary_size=0)
        self.node_data = []
        self.configure_model_creation(episodes_topic, episodes_msg)
        self.last_episode = Episode()

    def calculate_activation(self, perception=None, activation_list=None):
        filtered_activation_list = {node:activation for node, activation in activation_list.items() if activation["node_type"] != "WorldModel"}
        if filtered_activation_list:
            self.calculate_activation_max(filtered_activation_list)
        else:
            self.activation.activation=0.0
            self.activation.timestamp=self.get_clock().now().to_msg()

    def episode_callback(self, msg):
        episode = episode_msg_to_obj(msg)
        self.last_episode = episode
        if not episode.parent_policy: # Parent policy is empty if no specific Utility Model/Policy is being executed
            self.episodic_buffer.add_episode(episode)
            for goal, reward in episode.reward_list.items():
                if not isclose(reward, 0.0):
                    if goal in self.unlinked_drives:
                        drive = goal
                        goal = None
                        self.node_data.append(dict(node_type="Goal", drive=drive))
                        self.get_logger().info(f"Unlinked drive found: {drive}. Goal node to be created.")
                    if not self.linked_cnode(goal):
                        self.node_data.append(dict(node_type="UtilityModel", goal=goal, trace=deepcopy(self.episodic_buffer.main_buffer)))
                        self.episodic_buffer.clear()
                        self.get_logger().info(f"Goal {goal} not linked to any CNode. Utility Model to be created.")

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
        await self.create_models()
        self.last_episode.parent_policy = self.name
        self.last_episode.old_perception = self.last_episode.perception
        self.last_episode.action = Action()
        self.episode_publisher.publish(episode_obj_to_msg(self.last_episode))
        response.policy=self.name
        return response
    
    async def create_models(self):
        created_world_model = None
        created_goal = None
        drive = None
        if self.missing_world_model:
            self.get_logger().info("Missing World Model. Creating new World Model...")
            created_world_model = await self.create_world_model()
        if self.node_data:
            self.get_logger().info("Creating models for " + str(len(self.node_data)) + " nodes...")
            for node in self.node_data:
                node_type = node["node_type"]
                if node_type == "Goal":
                    drive = node["drive"]
                    self.get_logger().info("Creating Goal: " + drive)
                    created_goal = await self.create_goal(drive)
                elif node_type == "UtilityModel":
                    goal = node.get("goal", None)
                    if not goal and created_goal:
                        goal = created_goal
                    else:
                        raise ValueError("Goal is required for UtilityModel creation.")
                    if created_world_model:
                        world_model = created_world_model
                    else:
                        world_model = self.get_max_activation_node("WorldModel")
                    trace = node.get("trace", [])
                    self.get_logger().info("Creating Utility Model for goal: " + str(goal))
                    await self.create_utility_model(goal, drive, world_model, trace)
                else:
                    self.get_logger().warn("Unknown node type: " + node_type)
        self.node_data = []  # Clear node data after processing

    async def create_world_model(self):
        name = self.generate_node_name("WorldModel")
        classname = self.default_class.get("WorldModel", "cognitive_nodes.world_model.WorldModel") 
        creation_response = await self.create_node_client(name=name, class_name=classname, parameters=self.default_params.get("WorldModel", {}))
        if creation_response.created:
            self.get_logger().info(f"{classname}: {name} created successfully.")
            return name
        else:
            self.get_logger().error(f"Failed to create {classname}: {name}")
            return None
        
    async def create_goal(self, drive):
        name = self.generate_node_name("Goal")
        classname = self.default_class.get("Goal", "cognitive_nodes.goal.Goal")
        parameters = self.default_params.get("Goal", {})
        neighbors = {"neighbors": [{"name": drive, "node_type": "Drive"}]}
        creation_response = await self.create_node_client(name=name, class_name=classname, parameters={**parameters, **neighbors})
        if creation_response.created:
            self.get_logger().info(f"{classname}: {name} created successfully.")
            return name
        else:
            self.get_logger().error(f"Failed to create {classname}: {name}")
            return None
    
    async def create_utility_model(self, goal, drive, world_model, trace):
        utility_model_name = self.generate_node_name("UtilityModel")
        ident = f"{world_model}__{goal}__{utility_model_name}"
        space_class = self.default_class.get("Space")
        pnode_class = self.default_class.get("PNode")
        cnode_class = self.default_class.get("CNode")
        utility_model_class = self.default_class.get("UtilityModel")
        pnode_parameters = self.default_params.get("PNode", {})
        cnode_parameters = self.default_params.get("CNode", {})
        utility_model_parameters = self.default_params.get("UtilityModel", {})

        # Create P-Node
        pnode_name = f"pnode_{ident}"
        pnode_success = await self.create_node_client(name = pnode_name, class_name = pnode_class, parameters = {**pnode_parameters, "space_class": space_class})
        if not pnode_success.created:
            self.get_logger().error(f"Failed to create P-Node: {pnode_name}")

        pnode_points_service = f"pnode/{pnode_name}/add_points"
        if pnode_points_service not in self.node_clients:
            self.node_clients[pnode_points_service] = ServiceClientAsync(self, AddPoints, pnode_points_service, self.cbgroup_client)
        points = [perception_dict_to_msg(episode.old_perception) for episode in trace]
        confidences = list(np.ones(len(points)))
        pnode_points_response = await self.node_clients[pnode_points_service].send_request_async(points=points, confidences=confidences)
        if not pnode_points_response.added:
            self.get_logger().error(f"Failed to add points to P-Node: {pnode_name}")

        # Create C-Node
        cnode_name = f"cnode_{ident}"
        neighbor_dict = {world_model: "WorldModel", pnode_name: "PNode", goal: "Goal"}
        neighbors = {
            "neighbors": [{"name": node, "node_type": node_type} for node, node_type in neighbor_dict.items()]
        }
        cnode_creation_response = await self.create_node_client(name=cnode_name, class_name=cnode_class, parameters={**cnode_parameters, **neighbors})
        if not cnode_creation_response.created:
            self.get_logger().error(f"Failed to create C-Node: {cnode_name}")

        # Create Utility Model
        neighbors = {"neighbors": [{"name": cnode_name, "node_type": "CNode"}]}
        creation_response = await self.create_node_client(name=utility_model_name, class_name=utility_model_class, parameters={**utility_model_parameters, **neighbors})
        if not creation_response.created:
            self.get_logger().error(f"Failed to create UtilityModel: {utility_model_name}")
        utility_model_trace_service = f"utility_model/{utility_model_name}/add_trace"
        if utility_model_trace_service not in self.node_clients:
            self.node_clients[utility_model_trace_service] = ServiceClientAsync(self, AddTrace, utility_model_trace_service, self.cbgroup_client)
        if drive:
            reward_node = drive
        else:
            reward_node = goal
        rewards = [episode.reward_list[reward_node] for episode in trace]
        trace_success = await self.node_clients[utility_model_trace_service].send_request_async(episodes=episode_obj_list_to_msg_list(trace), rewards=rewards)
        if not trace_success.added:
            self.get_logger().error(f"Failed to add trace to UtilityModel: {utility_model_name}")

    def read_activation_callback(self, msg):
        super().read_activation_callback(msg)
        # Process world model activations
        updated_activations = all((self.activation_inputs[node_name]['updated'] for node_name in self.activation_inputs)) 
        self.get_logger().debug(f"Updated activations: {updated_activations}")
        if updated_activations:
            world_model_activations = [not isclose(activation["data"].activation, 0.0) for activation in self.activation_inputs.values() if activation["node_type"] == "WorldModel"]
            if not any(world_model_activations):
                self.get_logger().debug("No active World Model found. New World Model to be created.")
                self.missing_world_model = True
            else:
                self.get_logger().debug("Active World Model found.")
                self.missing_world_model = False