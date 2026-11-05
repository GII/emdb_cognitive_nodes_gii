import numpy as np
from math import isclose

from cognitive_nodes.robot_purpose import RobotPurpose
from cognitive_nodes.utils import LTMSubscription

from cognitive_node_interfaces.msg import Activation
from cognitive_node_interfaces.srv import IsSatisfied


def _clamp(value, lower, upper):
    return max(lower, min(upper, value))


class Competence(RobotPurpose, LTMSubscription):
    """
    Intrinsic motivation that tracks the competence of the robot in achieving its goals and drives it to improve them.

    The node listens to the LTM and adds every C-Node as a neighbor. From the activations of the C-Nodes it reads their
    ``competence`` and ``delta_competence`` metacognitive parameters. It gives each C-Node a score that is higher when its competence
    is lower and its delta competence is higher in absolute value, scales the scores to the interval
    [``min_activation``, activation of the node] (C-Nodes with a score of 0 keep 0), and publishes the result for each Goal
    connected to the C-Node on ``cognitive_node/<goal>/override_activation``
    (:class:`cognitive_node_interfaces.msg.Activation`).

    The node is satisfied when the competence of all the C-Nodes is 1.0 (this is also true if there is no C-Node).
    """

    def __init__(self, name='robot_purpose', class_name='cognitive_nodes.robot_purpose.RobotPurpose', node_type="RobotPurpose", weight=1, drive_id=None, purpose_type="Need", terminal=False, min_activation=0, ltm_id="", **params):
        """
        Constructor of the Competence class. Unlike :class:`cognitive_nodes.robot_purpose.RobotPurpose`, ``purpose_type`` defaults to "Need".

        :param name: The name of the node.
        :type name: str
        :param class_name: The name of the node class.
        :type class_name: str
        :param node_type: The type of the node, defaults to "RobotPurpose".
        :type node_type: str
        :param weight: The weight of the purpose. Defaults to 1.
        :type weight: float
        :param drive_id: The ID of the Drive node associated with the purpose.
        :type drive_id: str or None
        :param purpose_type: The type of the purpose (Need or Mission). Defaults to "Need".
        :type purpose_type: str
        :param terminal: Whether the purpose is terminal; it is returned as is in the ``get_satisfaction`` response. Defaults to False.
        :type terminal: bool
        :param min_activation: Lower bound of the override activations given to the goals with a positive score. Defaults to 0.
        :type min_activation: float
        :param ltm_id: Id of the LTM whose state the node follows.
        :type ltm_id: str
        :param params: Additional parameters passed to :class:`cognitive_nodes.robot_purpose.RobotPurpose`.
        :type params: dict
        """
        super().__init__(name, class_name, node_type, weight, drive_id, purpose_type, terminal, **params)
        self.configure_ltm_subscription(ltm_id, self.cbgroup_activation)
        self.cnode_metacognitive_params = {}
        self.goal_to_cnode_mapping = {}
        self.goal_activations = {}
        self.goal_override_publishers = {}
        self.override_activation_msg = Activation()
        self.override_activation_msg.node_name = self.name
        self.override_activation_msg.node_type = self.node_type
        self.min_activation = min_activation
        self.LTM_id = ltm_id

    def _extract_node_name(self, node_data):
        """
        Extract the goal name from an entry in the LTM dump.

        :param node_data: Goal information received from the LTM.
        :type node_data: dict
        :return: Goal name or None if it cannot be resolved.
        :rtype: str
        """
        if isinstance(node_data, dict):
            return node_data.get("name",  None)
        return None
    
    def _extract_goals_from_cnode(self, cnode_name, cnode_data, mapping_dict):
        """
        Extract the goals from a C-Node entry in the LTM dump and update the mapping dictionary.

        :param cnode_data: C-Node information received from the LTM.
        :type cnode_data: dict
        """
        neighbors = cnode_data.get("neighbors", [])
        goals = [neighbor.get("name") for neighbor in neighbors if neighbor.get("node_type") == "Goal"]
        for goal in goals:
            # Check if goal already exists in the mapping dictionary
            if goal in mapping_dict:
                self.get_logger().warn(f"Goal '{goal}' is already mapped to C-Node '{mapping_dict[goal]}'. Overwriting with C-Node '{cnode_name}'.")
            mapping_dict[goal] = cnode_name

    def _extract_cnode_metacognitive_params(self, activation_list):
        """
        Extract the metacognitive parameters used by the competence node from a goal entry.

        :param activation_list: List of activation values for the goal.
        :type activation_list: list
        :return: Dictionary with competence-related values.
        :rtype: dict
        """
        cnodes_activations = [activation_list[node]["data"] for node in activation_list if activation_list[node]["data"].node_type == "CNode"]
        for activation in cnodes_activations:
            cnode_name = activation.node_name
            metacognitive_params = self.read_metacognitive_parameters(activation)
            self.cnode_metacognitive_params[cnode_name] = metacognitive_params

    def _score_competence(self, competence, delta_competence):
        """
        Compute a modular score for a goal.

        Lower competence and higher delta competence should produce a higher score.

        :param competence: Goal competence value.
        :type competence: float
        :param delta_competence: Goal delta competence value.
        :type delta_competence: float
        :return: A normalized score in the [0, 1] range.
        :rtype: float
        """
        competence_score = 1.0 - _clamp(competence, 0.0, 1.0)
        delta = _clamp(abs(delta_competence), 0.0, 1.0)
        delta_score = 2*(delta / (1.0 + delta))

         # Provide a bit of score to C-nodes with no delta competence, to avoid them being ignored.
        if isclose(delta_score, 0.0) and not isclose(competence, 0.0):
            delta_score = 0.05

        return competence_score * delta_score

    def _scale_activations(self, goal_scores):
        """
        Normalize goal scores to the competence activation interval.

        Goals with score <= 0 keep activation 0.
        """
        upper_bound = max(0.0, float(self.activation.activation))
        lower_bound = min(max(0.0, float(self.min_activation)), upper_bound)

        if upper_bound <= 0.0:
            return {goal_name: 0.0 for goal_name in goal_scores}

        goal_names = np.array(list(goal_scores.keys()), dtype=object)
        scores = np.array(list(goal_scores.values()), dtype=float)

        positive_mask = scores > 0.0
        if not np.any(positive_mask):
            return {goal_name: 0.0 for goal_name in goal_scores}

        positive_scores = scores[positive_mask]
        min_score = positive_scores.min()
        max_score = positive_scores.max()

        activations = np.zeros_like(scores, dtype=float)

        if max_score == min_score:
            activations[positive_mask] = upper_bound
        else:
            normalized = (positive_scores - min_score) / (max_score - min_score)
            activations[positive_mask] = np.clip(
                lower_bound + normalized * (upper_bound - lower_bound),
                lower_bound,
                upper_bound,
            )

        return dict(zip(goal_names, activations.tolist()))

    async def read_ltm(self, ltm_dump):
        """
        Method that processes the LTM data. It adds as neighbors the C-Nodes that are new in the LTM and
        deletes those that are no longer there, and rebuilds the mapping from each Goal to its C-Node.
        This is a coroutine and must be awaited.

        :param ltm_dump: Dictionary with the data from the LTM.
        :type ltm_dump: dict
        """
        # Read existing goals
        previous_cnode_names = set(self.cnode_metacognitive_params.keys())

        # Read current C-Nodes from LTM
        cnodes = ltm_dump.get("CNode", [])
        cnode_list = []
        goal_mapping = {}
        for cnode, cnode_data in cnodes.items():
            self._extract_goals_from_cnode(cnode, cnode_data, goal_mapping)
            cnode_list.append(cnode)
        current_goal_names = set(cnode_list)

        # Update neighbors based on the difference between previous and current goals
        for cnode_name in previous_cnode_names - current_goal_names:
            await self.delete_neighbor_client(self.name, cnode_name)
            self.cnode_metacognitive_params.pop(cnode_name, None)

        for cnode_name in current_goal_names - previous_cnode_names:
            await self.add_neighbor_client(self.name, cnode_name)
            self.cnode_metacognitive_params[cnode_name] = {}
        self.goal_to_cnode_mapping = goal_mapping


    def calculate_override_activations(self, activation_list=None):
        """
        Calculate the override activations for each goal based on the competence of the robot in achieving them,
        and publish them on ``cognitive_node/<goal>/override_activation``. The activation of a Goal is that of its C-Node.

        :param activation_list: Not used.
        :type activation_list: dict
        """
        cnode_scores = {}

        for cnode_name, params in self.cnode_metacognitive_params.items():
            cnode_scores[cnode_name] = self._score_competence(
                params.get("competence", 0.0),
                params.get("delta_competence", 0.0),
            )

        cnode_activations = self._scale_activations(cnode_scores)

        self.override_activation_msg.timestamp = self.get_clock().now().to_msg()
        for goal_name, cnode_name in self.goal_to_cnode_mapping.items():
            goal_activation = cnode_activations.get(cnode_name, 0.0)
            self.goal_activations[goal_name] = goal_activation

            if goal_name not in self.goal_override_publishers:
                publisher = self.create_publisher(
                    Activation,
                    f"cognitive_node/{goal_name}/override_activation",
                    10,
                )
                self.goal_override_publishers[goal_name] = publisher

            self.override_activation_msg.activation = goal_activation
            self.goal_override_publishers[goal_name].publish(self.override_activation_msg)


    def calculate_activation(self, perception=None, activation_list=None):
        """
        Calculate the activation of the Competence node based on the competence of the robot in achieving its goals.
        Publishes override activations for each goal based on their competence and delta competence.
        The activation of the node itself is not changed. Nothing is returned.

        :param perception: Not used.
        :type perception: core.container.Container or None
        :param activation_list: Activations of the neighbor nodes (as kept in ``activation_inputs``), from which the metacognitive parameters of the C-Nodes are read. It is required in practice.
        :type activation_list: dict
        """
        super().calculate_activation(perception, activation_list)
        self._extract_cnode_metacognitive_params(activation_list)
        self.calculate_override_activations(activation_list)

    def get_satisfaction_callback(self, request:IsSatisfied.Request, response:IsSatisfied.Response):
        """
        Check if the robot purpose has been satisfied. Unlike :class:`cognitive_nodes.robot_purpose.RobotPurpose`, ``updated`` is always True.

        :param request: Request. Its timestamp is not used.
        :type request: cognitive_node_interfaces.srv.IsSatisfied.Request
        :param response: Response that indicates if the robot purpose is satisfied or not.
        :type response: cognitive_node_interfaces.srv.IsSatisfied.Response
        :return: Response that indicates if the robot purpose is satisfied or not.
        :rtype: cognitive_node_interfaces.srv.IsSatisfied.Response
        """
        self.get_logger().debug('Calculating satisfaction..')
        response.satisfied = self.calculate_satisfaction()
        response.purpose_type = self.purpose_type
        response.terminal = self.terminal
        response.updated = True
        return response

    def calculate_satisfaction(self):
        """
        Calculate whether the robot purpose is satisfied: the competence of every C-Node must be 1.0.

        :return: True if the robot purpose is satisfied, False otherwise.
        :rtype: bool
        """
        all_competence = np.array([params.get("competence", 0.0) for params in self.cnode_metacognitive_params.values()])
        satisfied = bool(np.all(np.isclose(np.ones_like(all_competence), all_competence)))
        return satisfied



