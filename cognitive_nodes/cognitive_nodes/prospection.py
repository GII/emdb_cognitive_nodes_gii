
from cognitive_nodes.drive import Drive
from cognitive_nodes.policy import Policy
from core.service_client import ServiceClient, ServiceClientAsync
from cognitive_nodes.utils import LTMSubscription

from cognitive_node_interfaces.msg import SuccessRate
from cognitive_node_interfaces.srv import ContainsSpace, SendSpace, GetKnowledge




class ProspectionDrive(Drive, LTMSubscription):
    def __init__(self, name="drive", class_name="cognitive_nodes.drive.Drive", ltm_id=None, min_pnode_rate=0.84, min_goal_rate=0.84,**params):
        super().__init__(name, class_name, **params)
        if ltm_id is None:
            raise Exception('No LTM input was provided.')
        else:    
            self.LTM_id = ltm_id
        self.goals_info=None
        self.min_pnode_rate=min_pnode_rate
        self.min_goal_rate=min_goal_rate
        self.configure_prospection_suscriptor(self.LTM_id)
        self.get_knowledge_service = self.create_service(GetKnowledge, 'drive/' + str(
            name) + '/get_knowledge', self.get_knowledge_callback, callback_group=self.cbgroup_server)
    
    def configure_prospection_suscriptor(self, ltm):
        self.configure_ltm_subscription(ltm)
        self.pnode_subscriptions = {}
        self.goal_subscriptions = {}
        self.learned_pnodes=[]
        self.learned_goals=[]
        self.pnode_goals_dict={} #Dictionary of the goals linked to each pnode {'pnode0': ['goal0' ... ], ..., 'pnodeN': ['goal', ... ]}
        self.found_knowledge={} #Dictionary of upstream goal relationship found {'goal0': ['goal1', 'goal2',....], ..., 'goalN': {...}}
        self.discarded_knowledge={} #Dictionary of upstream goal relationship found {'goal0': ['goal1', 'goal2',....], ..., 'goalN': {...}}
        self.new_knowledge=False

    def read_ltm(self, ltm_dump):
        pnodes = ltm_dump["PNode"]
        goals = ltm_dump["Goal"]
        self.goals_dict = goals
        for pnode in pnodes:
            if pnode not in self.pnode_subscriptions.keys():
                self.pnode_subscriptions[pnode] = self.create_subscription(SuccessRate, f"/pnode/{pnode}/success_rate", self.success_callback, 1, callback_group=self.cbgroup_activation)
        
        for goal in goals:
            if goal not in self.goal_subscriptions.keys():
                self.goal_subscriptions[goal] = self.create_subscription(SuccessRate, f"/goal/{goal}/confidence", self.success_callback, 1, callback_group=self.cbgroup_activation)
        
        changes = self.changes_in_pnodes(ltm_dump)
        if changes:
            self.pnode_goals_dict = self.find_goals(ltm_dump)


    def find_goals(self, ltm_dump): #TODO refactor with find_goals in effectance.py
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

    def changes_in_pnodes(self, ltm_dump): #TODO refactor with changes_in_pnodes in effectance.py
        """
        Returns True if a P-Node has been added or deleted
        """
        current_pnodes = set(self.pnode_goals_dict.keys())
        new_pnodes = set(ltm_dump["PNode"].keys())
        return not current_pnodes == new_pnodes
    
    async def success_callback(self, msg: SuccessRate):
        node_name=msg.node_name
        node_type=msg.node_type
        success_rate=msg.success_rate
        if node_type=="PNode":
            learned_list=self.learned_pnodes
            threshold=self.min_pnode_rate
        elif node_type=="Goal":
            learned_list=self.learned_goals
            threshold=self.min_goal_rate
        else:
            raise RuntimeError("Expected node type 'Pnode' or 'Goal'")
        if success_rate>threshold and node_name not in learned_list:
            learned_list.append(node_name)
            await self.do_prospection()

    async def do_prospection(self):
        self.get_logger().info(f"DEBUG - Performing prospection...")
        for goal in self.learned_goals:
            for pnode in self.learned_pnodes:
                #If a relationship has not been found before
                discovered = any(element in self.pnode_goals_dict[pnode] for element in self.found_knowledge.get(goal, []))
                discarded = any(element in self.pnode_goals_dict[pnode] for element in self.discarded_knowledge.get(goal, []))
                if not (discovered or discarded): 
                    self.get_logger().info(f"DEBUG - Searching relation between {goal} and {pnode}")
                    #Get goal space:
                    service_name = f"goal/{goal}/send_space"
                    if service_name not in self.node_clients:
                        self.node_clients[service_name] = ServiceClientAsync(self, SendSpace, service_name, self.cbgroup_client)
                    goal_space = await self.node_clients[service_name].send_request_async()
                    #Get pnode space:
                    service_name = f"pnode/{pnode}/send_space"
                    if service_name not in self.node_clients:
                        self.node_clients[service_name] = ServiceClientAsync(self, SendSpace, service_name, self.cbgroup_client)
                    pnode_space = await self.node_clients[service_name].send_request_async()
                    #Goal contains PNode
                    service_name = f"goal/{goal}/contains_space"
                    if service_name not in self.node_clients:
                        self.node_clients[service_name] = ServiceClientAsync(self, ContainsSpace, service_name, self.cbgroup_client)
                    pnode_in_goal = (await self.node_clients[service_name].send_request_async(
                        labels=pnode_space.labels, data=pnode_space.data, confidences=pnode_space.confidences
                        )).contained
                    #PNode contains Goal
                    service_name = f"pnode/{pnode}/contains_space"
                    if service_name not in self.node_clients:
                        self.node_clients[service_name] = ServiceClientAsync(self, ContainsSpace, service_name, self.cbgroup_client)
                    goal_in_pnode = (await self.node_clients[service_name].send_request_async(
                        labels=goal_space.labels, data=goal_space.data, confidences=goal_space.confidences
                        )).contained
                    #TODO THIS IS TESTING BOTH THAT THE GOAL IS INSIDE THE PNODE OR THE PNODE INSIDE THE GOAL. WE HAVE TO DECIDE THE 
                    #MOST APPROPRIATE METHOD TO DECIDE WHEN TO CHAIN OR NOT TO CHAIN GOALS.
                    #IF THE GOAL IS CONTAINED IN THE PNODE WE ARE SURE THAT THE GOALS MUST BE CHAINED.
                    #IF THE PNODE IS CONTAINED IN THE GOAL, THERE IS SOME PROBABILITY THAT ACHIEVING THE GOAL WILL ACTIVATE THE PNODE
                    if pnode_in_goal or goal_in_pnode:
                        self.get_logger().info(f"Found a relation between Goal {goal} and P-Node {pnode}")
                        if not any(self.has_loop(upstream_goal, goal) for upstream_goal in self.pnode_goals_dict[pnode]):
                            if not self.found_knowledge.get(goal, []):
                                self.found_knowledge[goal]=[]
                            linked_goals=self.pnode_goals_dict[pnode]
                            goal_neighbors=self.get_neighbor_names(goal)
                            self.get_logger().info(f"DEBUG - Downstream goal neighbors: {goal_neighbors}")
                            linked_goals_to_add=[linked_goal for linked_goal in linked_goals if linked_goal not in goal_neighbors]
                            linked_goals_to_discard=[linked_goal for linked_goal in linked_goals if linked_goal in goal_neighbors]
                            if linked_goals_to_add:
                                self.get_logger().info(f"DEBUG - Goals to add: {linked_goals_to_add}")
                                self.found_knowledge[goal].extend(linked_goals_to_add)
                                self.new_knowledge=True
                                self.get_logger().info(f"Found knowledge: {self.found_knowledge}")
                                return #Return after one link so that neighbors are connected one by one, to avoid creating a loop that goes unchecked by the has_loops() method
                            if linked_goals_to_discard:
                                if not self.discarded_knowledge.get(goal, []):
                                    self.discarded_knowledge[goal]=[]
                                    self.discarded_knowledge[goal].extend(self.pnode_goals_dict[pnode])
                        else:
                            if not self.discarded_knowledge.get(goal, []):
                                self.discarded_knowledge[goal]=[]
                            self.discarded_knowledge[goal].extend(self.pnode_goals_dict[pnode])
                            self.get_logger().warn("Goals will not be linked because it would create a loop!")
                    else:
                        if not self.discarded_knowledge.get(goal, []):
                            self.discarded_knowledge[goal]=[]
                        self.discarded_knowledge[goal].extend(self.pnode_goals_dict[pnode])

                else:
                    self.get_logger().info(f"DEBUG - Relationship between {goal} and {pnode} found before.")

    def traverse_neighbors(self, goal_name, downstream_goal, visited):
        """
        Traverse the neighbors of a goal to check if the downstream goal is in the chain.
        
        Args:
            goal_name (str): The name of the current goal being checked.
            downstream_goal (str): The name of the downstream goal to check for loops.
            visited (set): Set of already visited nodes to prevent revisiting.
        
        Returns:
            bool: True if the downstream goal is found in the chain, False otherwise.
        """
        if goal_name in visited:
            return False  # Prevent re-checking already visited nodes
        visited.add(goal_name)

        # Get neighbors of the current goal
        neighbors = self.goals_dict.get(goal_name, {}).get("neighbors", [])
        for neighbor in neighbors:
            # Check if the downstream goal is found in the chain
            if neighbor["name"] == downstream_goal:
                return True
            # Recur for each neighbor
            if self.traverse_neighbors(neighbor["name"], downstream_goal, visited):
                return True

        return False
    
    def get_neighbor_names(self, goal_name):
        neighbors = self.goals_dict.get(goal_name, {}).get("neighbors", [])
        names = [neighbor["name"] for neighbor in neighbors]
        return names

    def has_loop(self, upstream_goal, downstream_goal):
        """
        Checks if adding downstream_goal to upstream_goal's neighbors creates a loop.
        
        Args:
            upstream_goal (str): The name of the upstream goal.
            downstream_goal (str): The name of the downstream goal.
        
        Returns:
            bool: True if a loop would be created, False otherwise.
        """
        visited = set()
        return self.traverse_neighbors(upstream_goal, downstream_goal, visited)
    
    def has_neighbor(self, goal_name, neighbor_name):
        neighbors = self.goals_dict.get(goal_name, {}).get("neighbors", [])
        for neighbor in neighbors:
            # Check if the neighbor exists
            if neighbor["name"] == neighbor_name:
                return True
        return False


    def get_knowledge_callback(self, request, response):
        downstream_goals=[]
        upstream_goals=[]
        #Creates a flattened list of the found_knowledge dictionary
        self.get_logger().info(f"DEBUG - Found knowledge: {self.found_knowledge}")
        for ds_goal, linked_goals in self.found_knowledge.items():
            for us_goal in linked_goals:
                downstream_goals.append(ds_goal)
                upstream_goals.append(us_goal)
        self.get_logger().info(f"DEBUG - Downstream goals: {downstream_goals}, Upstream Goals: {upstream_goals}")
        self.new_knowledge=False
        response.downstream_goals=downstream_goals
        response.upstream_goals=upstream_goals
        return response
            
    def evaluate(self, perception=None):
        self.evaluation.evaluation=float(self.new_knowledge)
        self.evaluation.timestamp=self.get_clock().now().to_msg()

class PolicyProspection(Policy):
    def __init__(self, name='policy', class_name='cognitive_nodes.policy.Policy', ltm_id=None, drive_name=None, **params):
        super().__init__(name, class_name, **params)
        if ltm_id is None:
            raise RuntimeError('No LTM input was provided.')
        else:    
            self.LTM_id = ltm_id
        if drive_name is None:
            raise RuntimeError('No prospection drive was provided.')
        else:    
            self.drive = drive_name
        self.found_knowledge={}
        self.knowledge_client = ServiceClientAsync(self, GetKnowledge, f"drive/{self.drive}/get_knowledge", callback_group=self.cbgroup_client)

    async def execute_callback(self, request, response):
        self.get_logger().info('Executing policy: ' + self.name + '...')
        knowledge_msg = await self.knowledge_client.send_request_async()
        for ds_goal, us_goal in zip(knowledge_msg.downstream_goals, knowledge_msg.upstream_goals):
            if not self.found_knowledge.get(ds_goal, None):
                self.found_knowledge[ds_goal]=[]
            if us_goal not in self.found_knowledge[ds_goal]:
                #If knowledge not found previously, update neighbor and save relationship
                self.found_knowledge[ds_goal].append(us_goal)
                self.get_logger().info(f"Linking {us_goal} to {ds_goal}")
                await self.update_neighbor_client(ds_goal, us_goal, True)
        response.policy=self.name
        return response
        