import numpy as np

from cognitive_nodes.robot_purpose import RobotPurpose
from cognitive_nodes.utils import LTMSubscription

from cognitive_node_interfaces.msg import Activation
from cognitive_node_interfaces.srv import IsSatisfied


def _clamp(value, lower, upper):
    return max(lower, min(upper, value))


class Competence(RobotPurpose, LTMSubscription):
    """
    Intrinsic motivation that tracks the competence of the robot in achieving its goals and drives it to improve them.

    The node listens to the LTM and adds each eligible C-Node as a neighbor. A Goal is eligible when it has no Goal
    neighbors, which identifies the top-level goals in the current graph convention. From the activations of the C-Nodes it reads their
    ``competence`` and ``delta_competence`` metacognitive parameters. It gives each C-Node a score that is strongly driven by
    its competence deficit, amplified by positive competence change and strongly reduced by negative competence change,
    scales the scores to the interval [0, activation of the node], and publishes the result for each eligible Goal
    connected to the C-Node on ``cognitive_node/<goal>/override_activation``
    (:class:`cognitive_node_interfaces.msg.Activation`).

    The node is satisfied when the competence of all the C-Nodes is 1.0 (this is also true if there is no C-Node).
    """

    def __init__(
        self,
        name='robot_purpose',
        class_name='cognitive_nodes.robot_purpose.RobotPurpose',
        node_type="RobotPurpose",
        weight=1,
        drive_id=None,
        purpose_type="Need",
        terminal=False,
        min_activation=0,
        ltm_id="",
        competence_exponent=1.5,
        positive_delta_weight=1.0,
        delta_sample_count=10,
        competence_sample_count=10,
        minimum_activation_samples=10,
        **params,
    ):
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
        :param min_activation: Retained for configuration compatibility. Activations are now
            scaled absolutely so stable, highly competent goals can fade toward zero.
        :type min_activation: float
        :param ltm_id: Id of the LTM whose state the node follows.
        :type ltm_id: str
        :param competence_exponent: Exponent applied to the competence deficit.
            Values below 1 emphasize medium/high competence; values above 1 emphasize low competence.
        :type competence_exponent: float
        :param positive_delta_weight: Multiplier weight for positive competence changes.
        :type positive_delta_weight: float
        :param delta_sample_count: Number of observations needed for full delta-confidence.
        :type delta_sample_count: int
        :param competence_sample_count: Number of observations needed to trust raw competence.
        :type competence_sample_count: int
        :param minimum_activation_samples: Minimum observations before low competence can deactivate.
            This prevents sparse histories from ending motivation too early.
        :type minimum_activation_samples: int
        :param params: Additional parameters passed to :class:`cognitive_nodes.robot_purpose.RobotPurpose`.
        :type params: dict
        """
        super().__init__(name, class_name, node_type, weight, drive_id, purpose_type, terminal, **params)
        self.configure_ltm_subscription(ltm_id, self.cbgroup_activation)
        self.cnode_metacognitive_params = {}
        self.non_tracking_cnodes = set()
        self.goal_to_cnode_mapping = {}
        self.goal_activations = {}
        self.goal_override_publishers = {}
        self.override_activation_msg = Activation()
        self.override_activation_msg.node_name = self.name
        self.override_activation_msg.node_type = self.node_type
        self.LTM_id = ltm_id
        self.competence_exponent = max(0.0, float(competence_exponent))
        self.positive_delta_weight = max(0.0, float(positive_delta_weight))
        self.delta_sample_count = max(1, int(delta_sample_count))
        self.competence_sample_count = max(1, int(competence_sample_count))
        self.minimum_activation_samples = max(1, int(minimum_activation_samples))

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

    def _publish_goal_override(self, goal_name, activation):
        """
        Publish an override activation for a Goal.
        """
        if goal_name not in self.goal_override_publishers:
            self.goal_override_publishers[goal_name] = self.create_publisher(
                Activation,
                f"cognitive_node/{goal_name}/override_activation",
                10,
            )

        self.override_activation_msg.timestamp = self.get_clock().now().to_msg()
        self.override_activation_msg.activation = activation
        self.goal_override_publishers[goal_name].publish(self.override_activation_msg)

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
            if metacognitive_params.get("track_competence", True):
                self.non_tracking_cnodes.discard(cnode_name)
                self.cnode_metacognitive_params[cnode_name] = metacognitive_params
            else:
                self.non_tracking_cnodes.add(cnode_name)
                self.cnode_metacognitive_params.pop(cnode_name, None)

    def _score_competence(self, competence, delta_competence, sample_count=0):
        """
        Compute an absolute score for a goal.

        Lower competence and larger changes in competence should produce a higher score. The
        competence deficit is retained in the score so that a nearly mastered C-Node fades
        even when its recent competence changed.

        :param competence: Goal competence value.
        :type competence: float
        :param delta_competence: Goal delta competence value.
        :type delta_competence: float
        :return: A normalized score in the [0, 1] range.
        :rtype: float
        """
        competence_confidence = _clamp(
            int(sample_count) / float(self.competence_sample_count),
            0.0,
            1.0,
        )
        effective_competence = _clamp(competence, 0.0, 1.0) * competence_confidence
        deficit = (1.0 - effective_competence) ** self.competence_exponent
        delta = _clamp(delta_competence, -1.0, 1.0)

        # Dampen recent changes until enough observations are available. Low
        # competence still provides a strong score even for a brand-new C-Node.
        delta_confidence = _clamp(int(sample_count) / float(self.delta_sample_count), 0.0, 1.0)
        delta = delta * delta_confidence

        change_factor = _clamp(1.0 + self.positive_delta_weight * max(delta, 0.0), 0.0, 2.0)

        # Sparse success histories are not evidence of mastery. Keep a strong
        # deficit-based activation until enough observations exist.
        if int(sample_count) < self.minimum_activation_samples:
            deficit = max(deficit, 0.5)

        return _clamp(deficit * change_factor, 0.0, 1.0)

    def _scale_activations(self, goal_scores):
        """
        Scale absolute goal scores to the competence activation interval.

        Goals with score <= 0 keep activation 0.
        """
        upper_bound = max(0.0, float(self.activation.activation))

        if upper_bound <= 0.0:
            return {goal_name: 0.0 for goal_name in goal_scores}

        return {
            goal_name: upper_bound * _clamp(float(score), 0.0, 1.0)
            for goal_name, score in goal_scores.items()
        }

    async def read_ltm(self, ltm_dump):
        """
        Method that processes the LTM data. It adds as neighbors the C-Nodes that are new in the LTM and
        deletes those that are no longer there, and rebuilds the mapping from each Goal to its C-Node.
        This is a coroutine and must be awaited.

        :param ltm_dump: Dictionary with the data from the LTM.
        :type ltm_dump: dict
        """
        previous_goal_to_cnode_mapping = dict(self.goal_to_cnode_mapping)

        # Read existing goals
        previous_cnode_names = set(self.cnode_metacognitive_params.keys())

        # Read current C-Nodes from LTM. A Goal with no Goal neighbors is a
        # top-level target in the graph convention used by the architecture.
        goals = ltm_dump.get("Goal", {})
        eligible_goals = {
            goal_name
            for goal_name, goal_data in goals.items()
            if not any(neighbor.get("node_type") == "Goal"
                       for neighbor in goal_data.get("neighbors", []))
        }

        # Read current C-Nodes from LTM
        cnodes = ltm_dump.get("CNode", [])
        cnode_list = []
        goal_mapping = {}
        for cnode, cnode_data in cnodes.items():
            if cnode in self.non_tracking_cnodes:
                continue
            eligible_cnode_data = {
                **cnode_data,
                "neighbors": [
                    neighbor for neighbor in cnode_data.get("neighbors", [])
                    if neighbor.get("name") in eligible_goals
                ],
            }
            self._extract_goals_from_cnode(cnode, eligible_cnode_data, goal_mapping)
            cnode_list.append(cnode)
        current_cnode_names = {
            cnode_name for cnode_name in cnode_list
            if cnode_name in set(goal_mapping.values())
        }

        # Update neighbors based on the difference between previous and current goals
        for cnode_name in previous_cnode_names - current_cnode_names:
            await self.delete_neighbor_client(self.name, cnode_name)
            self.cnode_metacognitive_params.pop(cnode_name, None)

        for cnode_name in current_cnode_names - previous_cnode_names:
            await self.add_neighbor_client(self.name, cnode_name)
            self.cnode_metacognitive_params[cnode_name] = {}

        for goal_name in set(previous_goal_to_cnode_mapping) - set(goal_mapping):
            self._publish_goal_override(goal_name, 0.0)

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
                params.get("sample_count", 0),
            )

        cnode_activations = self._scale_activations(cnode_scores)

        self.override_activation_msg.timestamp = self.get_clock().now().to_msg()
        for goal_name, cnode_name in self.goal_to_cnode_mapping.items():
            goal_activation = cnode_activations.get(cnode_name, 0.0)
            self.goal_activations[goal_name] = goal_activation
            self._publish_goal_override(goal_name, goal_activation)


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
        Calculate whether the robot purpose is satisfied: the competence of every eligible
        top-level C-Node must be 1.0.

        :return: True if the robot purpose is satisfied, False otherwise.
        :rtype: bool
        """
        all_competence = np.array([
            params.get("competence", 0.0)
            for params in self.cnode_metacognitive_params.values()
        ])
        all_sample_counts = np.array([
            params.get("sample_count", 0)
            for params in self.cnode_metacognitive_params.values()
        ])
        competent = np.isclose(np.ones_like(all_competence), all_competence)
        enough_samples = all_sample_counts >= self.competence_sample_count
        satisfied = bool(np.all(competent & enough_samples))
        return satisfied
