"""Knowledge-reuse intrinsic motivation and its ROS-free test harness."""

import argparse
import asyncio
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

try:
    from cognitive_nodes.drive import Drive
    from cognitive_nodes.policy import Policy
    from cognitive_nodes.utils import LTMSubscription
    from core.container import Container
    from core_interfaces.srv import UpdateNeighbor
    from core.service_client import ServiceClientAsync
    from cognitive_node_interfaces.msg import SuccessRate
    from cognitive_node_interfaces.srv import (
        GetActivation,
        DuplicateNode,
        GetReusableKnowledge,
        SendSpace,
    )
except ModuleNotFoundError as error:
    if error.name not in {"rclpy", "cognitive_node_interfaces"}:
        raise

    class Drive:
        """Fallback base that keeps the schema harness ROS-free."""

    class Policy:
        """Fallback base that keeps the schema harness ROS-free."""

    class LTMSubscription:
        """Fallback mixin that keeps the schema harness ROS-free."""

    Container = None
    ServiceClientAsync = None
    GetReusableKnowledge = None
    SendSpace = None
    GetActivation = None
    DuplicateNode = None
    UpdateNeighbor = None
    SuccessRate = None


CHAIN_DEPTH_THRESHOLD = 3


# ---------------------------------------------------------------------------
# LTM schema helpers
# ---------------------------------------------------------------------------

def _nodes(ltm_dump, node_type):
    """Return a node-type section as ``{name: data}`` for either LTM shape."""
    section = (ltm_dump or {}).get(node_type, {})
    if isinstance(section, dict):
        return section
    if isinstance(section, list):
        return {
            node.get("name"): node
            for node in section
            if isinstance(node, dict) and node.get("name")
        }
    return {}


def _neighbor_names(node_data, node_type=None):
    """Extract neighbor names, optionally restricted to one node type."""
    if not isinstance(node_data, dict):
        return []
    neighbors = node_data.get("neighbors", [])
    if not isinstance(neighbors, list):
        return []
    return [
        neighbor["name"]
        for neighbor in neighbors
        if isinstance(neighbor, dict)
        and neighbor.get("name")
        and (node_type is None or neighbor.get("node_type") == node_type)
    ]


def _goal_depth(goal_name, downstream_goals, active_path=None):
    """Return the longest downstream goal chain rooted at ``goal_name``."""
    active_path = set() if active_path is None else active_path
    if goal_name in active_path:
        return 0
    next_path = active_path | {goal_name}
    downstream = downstream_goals.get(goal_name, ())
    return 1 + max(
        (_goal_depth(goal, downstream_goals, next_path) for goal in downstream),
        default=0,
    )


def build_goal_chains(ltm_dump):
    """Build the public knowledge-reuse schema from an LTM state dump.

    Each field in a drive's schema is aligned by index: ``goals``,
    ``domains``, ``policies``, ``pnodes``, and ``depth`` describe the same
    contextual chain entry.
    """
    goals = _nodes(ltm_dump, "Goal")
    cnodes = _nodes(ltm_dump, "CNode")
    policies = _nodes(ltm_dump, "Policy")

    upstream_goals = {
        goal_name: _neighbor_names(goal_data, "Goal")
        for goal_name, goal_data in goals.items()
    }
    downstream_goals = defaultdict(list)
    for goal_name, upstream in upstream_goals.items():
        for upstream_goal in upstream:
            downstream_goals[upstream_goal].append(goal_name)
    depths = {
        goal_name: _goal_depth(goal_name, downstream_goals)
        for goal_name in goals
    }

    policies_by_cnode = defaultdict(list)
    for policy_name, policy_data in policies.items():
        for cnode_name in _neighbor_names(policy_data, "CNode"):
            policies_by_cnode[cnode_name].append(policy_name)

    chains_by_drive = defaultdict(
        lambda: {
            "goals": [],
            "domains": [],
            "policies": [],
            "pnodes": [],
            "depth": [],
        }
    )
    seen = set()
    for cnode_name, cnode_data in cnodes.items():
        cnode_goals = _neighbor_names(cnode_data, "Goal")
        domains = _neighbor_names(cnode_data, "WorldModel")
        pnodes = _neighbor_names(cnode_data, "PNode")
        cnode_policies = _neighbor_names(cnode_data, "Policy")
        cnode_policies.extend(policies_by_cnode.get(cnode_name, ()))

        for goal_name in cnode_goals:
            goal_drives = _neighbor_names(goals.get(goal_name), "Drive")
            if not goal_drives:
                continue
            for drive_name in goal_drives:
                for domain in domains or [None]:
                    for policy_name in cnode_policies or [None]:
                        entry_key = (drive_name, goal_name, domain, policy_name)
                        if entry_key in seen:
                            continue
                        seen.add(entry_key)
                        chains = chains_by_drive[drive_name]
                        chains["goals"].append(goal_name)
                        chains["domains"].append(domain)
                        chains["policies"].append(policy_name)
                        chains["pnodes"].append(pnodes[0] if pnodes else None)
                        chains["depth"].append(depths.get(goal_name, 1))

    return dict(chains_by_drive)


def _find_candidates(goal_chains, new_goals, depth_threshold):
    """Find cross-domain goal pairs sharing a policy and a deep source chain."""
    candidates = []
    for drive_name, chains in goal_chains.items():
        entries = list(
            zip(
                chains["goals"],
                chains["domains"],
                chains["policies"],
                chains["pnodes"],
                chains["depth"],
            )
        )
        for goal, domain, policy, pnode, depth in entries:
            if goal not in new_goals or policy is None:
                continue
            for (
                candidate_goal,
                candidate_domain,
                candidate_policy,
                candidate_pnode,
                candidate_depth,
            ) in entries:
                if (
                    candidate_goal != goal
                    and candidate_policy == policy
                    and candidate_domain != domain
                    and depth - candidate_depth >= depth_threshold
                ):
                    candidates.append(
                        {
                            "drive": drive_name,
                            "goal": goal,
                            "domain": domain,
                            "policy": policy,
                            "pnode": pnode,
                            "candidate_goal": candidate_goal,
                            "candidate_domain": candidate_domain,
                            "candidate_pnode": candidate_pnode,
                            "candidate_depth": candidate_depth,
                        }
                    )
    return candidates


# ---------------------------------------------------------------------------
# Knowledge-reuse drive
# ---------------------------------------------------------------------------

class DriveKnowledgeReuse(Drive, LTMSubscription):
    """Drive activated when a newly observed goal can reuse a deep chain."""

    def __init__(
        self,
        name="knowledge_reuse_drive",
        class_name="cognitive_nodes.drive.Drive",
        ltm_id=None,
        depth_threshold=CHAIN_DEPTH_THRESHOLD,
        min_points=1,
        **params,
    ):
        super().__init__(name, class_name, **params)
        if ltm_id is None:
            raise ValueError("No LTM input was provided.")
        if depth_threshold < 0:
            raise ValueError("depth_threshold must be non-negative.")
        if min_points < 1:
            raise ValueError("min_points must be at least one.")

        self.LTM_id = ltm_id
        self.depth_threshold = depth_threshold
        self.min_points = min_points
        self.goal_chains = {}
        self.reuse_candidates = []
        self.reusable_knowledge = {"instructions": {}, "candidates": []}
        self._ltm_dump = {}
        self._known_goals = set()
        self._ltm_signature = None
        self._candidate_cache = []
        self._space_cache = {}
        self._score_cache = {}
        self._pending_pnode_subscriptions = {}
        self._pending_pnode_candidates = set()
        self.get_reusable_knowledge_service = self.create_service(
            GetReusableKnowledge,
            f"drive/{name}/get_reusable_knowledge",
            self.get_reusable_knowledge_callback,
            callback_group=self.cbgroup_server,
        )
        self.configure_ltm_subscription(ltm_id, self.cbgroup_activation)

    async def read_ltm(self, ltm_dump):
        """Refresh the schema and identify newly eligible reuse candidates."""
        goal_chains = build_goal_chains(ltm_dump)
        signature = self._knowledge_signature(ltm_dump, goal_chains)
        if signature == self._ltm_signature:
            self.get_logger().debug("Knowledge reuse: unchanged LTM state; skipping refresh.")
            return
        self._ltm_signature = signature
        self._ltm_dump = ltm_dump
        self.goal_chains = goal_chains
        current_goals = {
            goal
            for chains in self.goal_chains.values()
            for goal in chains["goals"]
        }
        self._known_goals = current_goals
        # Re-evaluate all contexts so space scores and YAML instructions stay
        # current even when an existing chain or P-Node changes.
        candidates = self._find_candidates(current_goals)
        self._candidate_cache = candidates
        self.get_logger().info(
            f"Knowledge reuse: LTM refresh found {len(current_goals)} goals and "
            f"{len(candidates)} reuse candidates."
        )
        active_targets = {
            candidate["candidate_pnode"]
            for candidate in candidates
            if candidate["candidate_pnode"]
        }
        for pnode_name in list(self._pending_pnode_candidates - active_targets):
            self._pending_pnode_candidates.discard(pnode_name)
            self._unsubscribe_pending_pnode(pnode_name)
        self.reusable_knowledge = await self._select_reusable_knowledge(candidates)
        self.reuse_candidates = self.reusable_knowledge["candidates"]
        self.get_logger().info(
            "Knowledge reuse: selected "
            f"{len(self.reuse_candidates)} candidates and generated "
            f"{sum(len(nodes) for nodes in self.reusable_knowledge['instructions'].values())} "
            "instructions."
        )

    @staticmethod
    def _knowledge_signature(ltm_dump, goal_chains):
        """Build a stable signature for topology relevant to reuse."""
        return (
            yaml.safe_dump(
                {
                    "Goal": _nodes(ltm_dump, "Goal"),
                    "CNode": _nodes(ltm_dump, "CNode"),
                    "Policy": _nodes(ltm_dump, "Policy"),
                    "WorldModel": _nodes(ltm_dump, "WorldModel"),
                    "PNode": _nodes(ltm_dump, "PNode"),
                    "chains": goal_chains,
                },
                sort_keys=True,
            )
        )

    def _subscribe_pending_pnode(self, pnode_name):
        if pnode_name in self._pending_pnode_subscriptions:
            return
        self.get_logger().info(
            f"Knowledge reuse: waiting for more points from candidate P-Node "
            f"{pnode_name}."
        )
        self._pending_pnode_subscriptions[pnode_name] = self.create_subscription(
            SuccessRate,
            f"/pnode/{pnode_name}/success_rate",
            self._pending_pnode_success_callback,
            1,
            callback_group=self.cbgroup_activation,
        )

    def _unsubscribe_pending_pnode(self, pnode_name):
        subscription = self._pending_pnode_subscriptions.pop(pnode_name, None)
        if subscription is not None:
            self.destroy_subscription(subscription)
            self.get_logger().info(
                f"Knowledge reuse: candidate P-Node {pnode_name} reached the "
                "minimum point requirement."
            )

    async def _pending_pnode_success_callback(self, msg):
        pnode_name = getattr(msg, "node_name", None)
        if pnode_name not in self._pending_pnode_candidates:
            return
        self.get_logger().debug(
            f"Knowledge reuse: received success-rate update for pending "
            f"P-Node {pnode_name}; retrying candidate selection."
        )
        self._space_cache.pop(pnode_name, None)
        self._score_cache = {
            key: value
            for key, value in self._score_cache.items()
            if key[3] != pnode_name
        }
        candidates = self._candidate_cache
        self.reusable_knowledge = await self._select_reusable_knowledge(candidates)
        self.reuse_candidates = self.reusable_knowledge["candidates"]

    async def _request_space(self, pnode_name):
        service_name = f"pnode/{pnode_name}/send_space"
        if service_name not in self.node_clients:
            self.node_clients[service_name] = ServiceClientAsync(
                self, SendSpace, service_name, self.cbgroup_client
            )
        response = await self.node_clients[service_name].send_request_async()
        return Container.from_msg(response.space)

    async def _request_activations(self, pnode_name, space):
        """Evaluate candidate points with the mature P-Node model."""
        service_name = f"cognitive_node/{pnode_name}/get_activation"
        if service_name not in self.node_clients:
            self.node_clients[service_name] = ServiceClientAsync(
                self, GetActivation, service_name, self.cbgroup_client
            )
        response = await self.node_clients[service_name].send_request_async(
            perception=space.to_msg()
        )
        return np.asarray(response.activation, dtype=float).reshape(-1)

    @staticmethod
    def _activation_mse(activations, memberships):
        """Compare mature-model activations with candidate memberships in [0, 1]."""
        activations = np.asarray(activations, dtype=float).reshape(-1)
        expected = np.clip(
            (np.asarray(memberships, dtype=float).reshape(-1) + 1.0) / 2.0,
            0.0,
            1.0,
        )
        if activations.size == 0 or activations.size != expected.size:
            return float("inf")
        return float(np.mean((activations - expected) ** 2))

    async def _select_reusable_knowledge(self, candidates):
        """Return finalized instructions and the best source per target goal.

        ``goal`` identifies the deeper source chain to copy, while
        ``candidate_goal`` identifies the shallow target chain receiving the
        duplicated knowledge.
        """
        selected = {}
        for candidate in candidates:
            target = candidate["candidate_pnode"]
            mature = candidate["pnode"]
            if not target or not mature:
                continue
            try:
                if target not in self._space_cache:
                    self.get_logger().debug(
                        f"Knowledge reuse: requesting space for candidate P-Node {target}."
                    )
                    self._space_cache[target] = await self._request_space(target)
                else:
                    self.get_logger().debug(
                        f"Knowledge reuse: using cached space for candidate P-Node {target}."
                    )
                target_space = self._space_cache[target]
            except (RuntimeError, ValueError) as error:
                self.get_logger().error(
                    f"Failed to read P-Node spaces for knowledge reuse: {error}"
                )
                continue
            if target_space is None:
                continue
            if target_space.size < self.min_points:
                self.get_logger().debug(
                    f"Knowledge reuse: candidate P-Node {target} has "
                    f"{target_space.size}/{self.min_points} points."
                )
                self._pending_pnode_candidates.add(target)
                self._subscribe_pending_pnode(target)
                continue
            self._pending_pnode_candidates.discard(target)
            self._unsubscribe_pending_pnode(target)
            space_signature = self._space_signature(target_space)
            score_key = (
                candidate["goal"],
                candidate["candidate_goal"],
                mature,
                target,
                space_signature,
            )
            if score_key in self._score_cache:
                scored_candidate = self._score_cache[score_key]
                self.get_logger().debug(
                    f"Knowledge reuse: using cached score for mature P-Node "
                    f"{mature} and candidate P-Node {target}: "
                    f"{scored_candidate['error']:.6f}."
                )
                target_goal = scored_candidate["candidate_goal"]
                current = selected.get(target_goal)
                if current is None or scored_candidate["error"] < current["error"]:
                    selected[target_goal] = scored_candidate
                continue
            try:
                activations = await self._request_activations(mature, target_space)
            except (RuntimeError, ValueError) as error:
                self.get_logger().error(
                    f"Failed to evaluate candidate points with mature P-Node "
                    f"{mature}: {error}"
                )
                continue
            scored_candidate = dict(candidate)
            scored_candidate["error"] = self._activation_mse(
                activations, target_space.memberships
            )
            if not np.isfinite(scored_candidate["error"]):
                self.get_logger().warning(
                    f"Knowledge reuse: discarded non-finite score for mature "
                    f"P-Node {mature} and candidate P-Node {target}."
                )
                continue
            self.get_logger().info(
                f"Knowledge reuse: compared mature P-Node {mature} with "
                f"candidate P-Node {target}; MSE={scored_candidate['error']:.6f}."
            )
            self._score_cache[score_key] = scored_candidate
            target_goal = scored_candidate["candidate_goal"]
            current = selected.get(target_goal)
            if current is None or scored_candidate["error"] < current["error"]:
                if current is not None:
                    self.get_logger().debug(
                        f"Knowledge reuse: replacing source goal "
                        f"{current['goal']} for target goal {target_goal}; "
                        f"MSE improved from {current['error']:.6f} to "
                        f"{scored_candidate['error']:.6f}."
                    )
                selected[target_goal] = scored_candidate
        self.get_logger().info(
            f"Knowledge reuse: finalized {len(selected)} target goal selections."
        )
        return self._build_instructions(selected.values())

    @staticmethod
    def _space_signature(space):
        """Identify the point data used for a cached mature-model score."""
        members = np.asarray(space.members, dtype=float)
        memberships = np.asarray(space.memberships, dtype=float)
        return (
            int(space.size),
            members.tobytes(),
            memberships.tobytes(),
        )

    def _build_instructions(self, selected_candidates):
        """Serialize selected source chains using experiment YAML conventions."""
        instructions = {"Goal": [], "CNode": [], "PNode": [], "Policy": []}
        policies = _nodes(self._ltm_dump, "Policy")
        selected_candidates = list(selected_candidates)
        for candidate in selected_candidates:
            self.get_logger().debug(
                f"Knowledge reuse: generating instructions by copying the chain "
                f"from source goal {candidate['goal']} to target goal "
                f"{candidate['candidate_goal']}."
            )
            first_element = True
            for goal_name in self._downstream_chain(candidate["goal"]):
                if first_element:
                    first_element = False
                    continue
                goal_data = _nodes(self._ltm_dump, "Goal").get(goal_name, {})
                self._add_instruction(instructions, "Goal", goal_name, goal_data)
                for cnode_name, cnode_data in self._context_cnodes(goal_name):
                    self._add_instruction(
                        instructions, "CNode", cnode_name, cnode_data
                    )
                    for policy_name, policy_data in policies.items():
                        if cnode_name in _neighbor_names(policy_data, "CNode"):
                            self._add_instruction(
                                instructions, "Policy", policy_name, policy_data
                            )
                    for pnode_name in _neighbor_names(cnode_data, "PNode"):
                        pnode_data = _nodes(self._ltm_dump, "PNode").get(
                            pnode_name, {}
                        )
                        self._add_instruction(
                            instructions, "PNode", pnode_name, pnode_data
                        )
        return {"instructions": instructions, "candidates": selected_candidates}

    def _downstream_chain(self, root_goal):
        """Return all downstream goals, including the root, without cycles."""
        upstream_goals = {
            name: _neighbor_names(data, "Goal")
            for name, data in _nodes(self._ltm_dump, "Goal").items()
        }
        downstream_goals = defaultdict(list)
        for goal_name, upstream in upstream_goals.items():
            for upstream_goal in upstream:
                downstream_goals[upstream_goal].append(goal_name)

        chain = []
        visited = set()

        def visit(goal_name):
            if goal_name in visited:
                return
            visited.add(goal_name)
            chain.append(goal_name)
            for downstream_goal in downstream_goals.get(goal_name, ()):
                visit(downstream_goal)

        visit(root_goal)
        return chain

    def _context_cnodes(self, goal_name):
        return [
            (cnode_name, cnode_data)
            for cnode_name, cnode_data in _nodes(self._ltm_dump, "CNode").items()
            if goal_name in _neighbor_names(cnode_data, "Goal")
        ]

    @staticmethod
    def _add_instruction(instructions, node_type, name, data):
        if any(item["name"] == name for item in instructions[node_type]):
            return
        instructions[node_type].append(
            {
                "name": name,
                "class_name": data.get("class_name", ""),
                "parameters": {"neighbors": data.get("neighbors", [])},
            }
        )

    def get_reusable_knowledge_callback(self, request, response):
        response.instructions = yaml.safe_dump(
            self.reusable_knowledge,
            default_flow_style=False,
            sort_keys=False,
        )
        return response

    def _find_candidates(self, new_goals):
        return _find_candidates(
            self.goal_chains,
            new_goals,
            self.depth_threshold,
        )

    def evaluate(self, perception=None):
        """Activate the drive when at least one reuse candidate is available."""
        self.evaluation.evaluation = float(bool(self.reuse_candidates))
        self.evaluation.timestamp = self.get_clock().now().to_msg()
        return self.evaluation


# Keep the shorter name available for configurations that name drives by their
# intrinsic motivation instead of by their implementation detail.
KnowledgeReuseDrive = DriveKnowledgeReuse


# ---------------------------------------------------------------------------
# Knowledge-reuse policy
# ---------------------------------------------------------------------------

class PolicyKnowledgeReuse(Policy):
    """Duplicate and reconnect the node chain returned by the reuse drive."""

    _DUPLICABLE_TYPES = frozenset({"Goal", "PNode", "CNode"})

    def __init__(
        self,
        name="policy_knowledge_reuse",
        class_name="cognitive_nodes.policy.Policy",
        ltm_id=None,
        drive_name=None,
        **params,
    ):
        if ltm_id is None:
            raise ValueError("No LTM input was provided.")
        if drive_name is None:
            raise ValueError("No knowledge-reuse drive was provided.")
        super().__init__(name, class_name, ltm_id=ltm_id, **params)
        self.LTM_id = ltm_id
        self.drive_name = drive_name
        self.knowledge_client = ServiceClientAsync(
            self,
            GetReusableKnowledge,
            f"drive/{drive_name}/get_reusable_knowledge",
            callback_group=self.cbgroup_client,
        )
        self._neighbor_client = ServiceClientAsync(
            self,
            UpdateNeighbor,
            f"{ltm_id}/update_neighbor",
            self.cbgroup_client,
        )
        self.aliases = {}

    async def execute_callback(self, request, response):
        """Duplicate the reusable chain and restore its aliased neighbors."""
        self.get_logger().info(f"Executing policy: {self.name}...")
        knowledge_response = await self.knowledge_client.send_request_async()
        try:
            knowledge = yaml.safe_load(knowledge_response.instructions) or {}
        except yaml.YAMLError as error:
            self.get_logger().error(
                f"Knowledge reuse instructions are not valid YAML: {error}"
            )
            response.policy = self.name
            return response

        if not isinstance(knowledge, dict):
            self.get_logger().error(
                "Knowledge reuse instructions must contain a mapping."
            )
            response.policy = self.name
            return response
        instructions = knowledge.get("instructions", {})
        self.aliases = await self._duplicate_nodes(instructions)
        await self._restore_neighbors(instructions, self.aliases)
        response.policy = self.name
        return response

    async def _duplicate_nodes(self, instructions):
        aliases = {}
        for node_type in ("Goal", "PNode", "CNode"):
            for instruction in instructions.get(node_type, []):
                source_name = instruction.get("name")
                if not source_name:
                    self.get_logger().error(
                        f"Knowledge reuse instruction for {node_type} has no name."
                    )
                    continue
                duplicate_service = (
                    f"cognitive_node/{source_name}/duplicate_node"
                )
                try:
                    client = ServiceClientAsync(
                        self,
                        DuplicateNode,
                        duplicate_service,
                        self.cbgroup_client,
                    )
                    result = await client.send_request_async(
                        name="",
                        include_neighbors=False,
                    )
                except (RuntimeError, ValueError) as error:
                    self.get_logger().error(
                        f"Failed to duplicate {node_type} {source_name}: {error}"
                    )
                    continue
                if not result.duplicated:
                    self.get_logger().error(
                        f"Node duplication failed for {node_type} {source_name}."
                    )
                    continue
                aliases[source_name] = result.duplicate_node_name
        return aliases

    async def _restore_neighbors(self, instructions, aliases):
        for node_type in ("Goal", "PNode", "CNode"):
            for instruction in instructions.get(node_type, []):
                source_name = instruction.get("name")
                duplicate_name = aliases.get(source_name)
                if not duplicate_name:
                    continue
                parameters = instruction.get("parameters", {})
                neighbors = parameters.get("neighbors", [])
                for neighbor in neighbors:
                    if not isinstance(neighbor, dict):
                        self.get_logger().error(
                            f"Invalid neighbor in instruction for {source_name}."
                        )
                        continue
                    neighbor_name = neighbor.get("name")
                    neighbor_type = neighbor.get("node_type")
                    if not neighbor_name or not neighbor_type:
                        self.get_logger().error(
                            f"Incomplete neighbor in instruction for {source_name}."
                        )
                        continue
                    aliased_neighbor = aliases.get(neighbor_name, neighbor_name)
                    result = await self._neighbor_client.send_request_async(
                        node_name=duplicate_name,
                        neighbor_name=aliased_neighbor,
                        operation=True,
                    )
                    if not result.success:
                        self.get_logger().error(
                            f"Failed to link {duplicate_name} to "
                            f"{aliased_neighbor}."
                        )


# ---------------------------------------------------------------------------
# ROS-free debug harness
# ---------------------------------------------------------------------------

class DummyKnowledgeReuseNode:
    """Small stand-in for the drive's LTM state handling."""

    def __init__(self, ltm_dump, depth_threshold=CHAIN_DEPTH_THRESHOLD):
        self.depth_threshold = depth_threshold
        self.goal_chains = build_goal_chains(ltm_dump)
        self.reuse_candidates = self._find_candidates()

    def _find_candidates(self):
        goals = {
            goal
            for chains in self.goal_chains.values()
            for goal in chains["goals"]
        }
        return _find_candidates(
            self.goal_chains,
            goals,
            self.depth_threshold,
        )


class DummyNode:
    """Logger provider used when loading real spaces without ROS."""

    def get_logger(self):
        class Logger:
            def debug(self, msg):
                pass

            def info(self, msg):
                print(f"INFO: {msg}")

            def warn(self, msg):
                print(f"WARNING: {msg}")

            warning = warn

            def error(self, msg):
                print(f"ERROR: {msg}")

            def fatal(self, msg):
                print(f"FATAL: {msg}")

        return Logger()


class KnowledgeReuseComparisonTest(DriveKnowledgeReuse):
    """Run production knowledge-reuse logic against real P-Node models."""
    from core.container import Container

    def __init__(self, ltm_dump, pnode_spaces, mature_pnodes, min_points=1):
        self._ltm_dump = ltm_dump
        self._spaces = pnode_spaces
        self._mature_pnodes = mature_pnodes
        self.min_points = min_points
        self.depth_threshold = CHAIN_DEPTH_THRESHOLD
        self.goal_chains = build_goal_chains(ltm_dump)
        self.get_logger = DummyNode().get_logger
        self._space_cache = {}
        self._score_cache = {}
        self._pending_pnode_candidates = set()
        self._pending_pnode_subscriptions = {}

    async def _request_space(self, pnode_name):
        return self._spaces[pnode_name]

    async def _request_activations(self, pnode_name, space):
        points = space._data
        return self._mature_pnodes[pnode_name].get_probability(points).reshape(-1)

    def _subscribe_pending_pnode(self, pnode_name):
        """Keep the fixture harness independent of ROS subscriptions."""
        self._pending_pnode_subscriptions[pnode_name] = None

    def _unsubscribe_pending_pnode(self, pnode_name):
        self._pending_pnode_subscriptions.pop(pnode_name, None)

    async def compare(self, candidate_goals):
        """Use the same candidate and selection methods as the live drive."""
        candidates = [
            candidate
            for candidate in self._find_candidates(candidate_goals)
            if candidate["candidate_pnode"] in self._spaces
            and candidate["pnode"] in self._mature_pnodes
        ]
        return await self._select_reusable_knowledge(candidates)


def _load_candidate_spaces(path, pnode_names, min_points, logger):
    """Load only the first candidate rows needed from the TSV export."""
    import pandas as pd
    from core.container import Container
    from cognitive_nodes.space import ANNSpace

    header = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    usecols = [column for column in header if column not in {"Iteration"}]
    rows_by_pnode = {}
    for chunk in pd.read_csv(path, sep="\t", usecols=usecols, chunksize=10000):
        for name in pnode_names - rows_by_pnode.keys():
            rows = chunk[chunk["Ident"] == name].head(min_points)
            if not rows.empty:
                rows_by_pnode[name] = rows
        if rows_by_pnode.keys() >= pnode_names:
            break

    spaces = {}
    for name, rows in rows_by_pnode.items():
        feature_labels = [
            label for label in rows.columns if label not in {"Ident", "confidence"}
        ]
        values = rows[feature_labels + ["confidence"]].to_numpy(dtype=float)
        data = Container(
            name=f"{name}_data",
            max_size=len(values),
            container_type="space",
            labels=feature_labels + ["confidence"],
        )
        data.push(values, timestamps=np.full(len(values), 0.0), src_labels=feature_labels + ["confidence"])
        spaces[name] = ANNSpace.populate_space(data, logger=logger, device="cpu")
    return spaces


def _load_mature_pnodes(fixture_dir, logger):
    import tempfile
    import torch
    import pandas as pd
    from cognitive_nodes.space import ANNSpace

    columns = pd.read_csv(
        fixture_dir / "pnodes_content_0.txt",
        sep="\t",
        nrows=0,
    ).columns.tolist()
    input_labels = [
        label for label in columns if label not in {"Iteration", "Ident", "confidence"}
    ]
    models = {}
    for model_file in fixture_dir.glob("pnode_*.pth"):
        name = model_file.stem
        checkpoint = torch.load(model_file, map_location="cpu")
        checkpoint["input_labels"] = input_labels
        temporary_model = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
        temporary_model.close()
        torch.save(checkpoint, temporary_model.name)
        models[name] = ANNSpace(
            ident=name,
            model_file=temporary_model.name,
            logger=logger,
            device="cpu",
        )
        Path(temporary_model.name).unlink()
    return models


async def comparison_main_async(fixture_dir, max_points=25):
    """Run the ROS-free P-Node comparison and instruction-generation test."""
    fixture_dir = Path(fixture_dir)
    with (fixture_dir / "neighbors_full_test.yaml").open(encoding="utf-8") as fixture:
        ltm_dump = yaml.safe_load(fixture)
    logger = DummyNode().get_logger()
    mature_pnodes = _load_mature_pnodes(fixture_dir, logger)
    candidate_names = {
        pnode
        for chains in build_goal_chains(ltm_dump).values()
        for pnode in chains["pnodes"]
        if pnode
    }
    spaces = _load_candidate_spaces(
        fixture_dir / "pnodes_content_0.txt",
        candidate_names,
        max_points,
        logger,
    )
    test = KnowledgeReuseComparisonTest(
        ltm_dump,
        spaces,
        mature_pnodes,
        min_points=max_points,
    )
    candidate_goals = {
        goal
        for chains in test.goal_chains.values()
        for goal in chains["goals"]
    }
    result = await test.compare(candidate_goals)
    print(f"Loaded mature P-Nodes: {len(mature_pnodes)}")
    print(f"Loaded candidate spaces: {len(spaces)}")
    print(f"Selected candidates: {len(result['candidates'])}")
    # Save test yaml instructions to a file for inspection if needed.
    print(f"Saving reusable knowledge instructions to: {'reusable_knowledge_test.yaml'}")
    with open("reusable_knowledge_test.yaml","w", encoding="utf-8") as output:
        yaml.safe_dump(result, output, default_flow_style=False, sort_keys=False)


def comparison_main():
    parser = argparse.ArgumentParser(description="Run the ROS-free P-Node comparison test.")
    parser.add_argument(
        "fixture_dir",
        nargs="?",
        type=Path,
        default=Path(__file__).parents[1] / "test" / "fixtures",
    )
    parser.add_argument("--max-points", type=int, default=25)
    args = parser.parse_args()
    asyncio.run(comparison_main_async(args.fixture_dir, args.max_points))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "fixture",
        nargs="?",
        type=Path,
        default=Path(__file__).parents[1] / "test" / "fixtures" / "neighbors_full_0.yaml",
        help="LTM-shaped YAML fixture",
    )
    args = parser.parse_args()

    with args.fixture.open(encoding="utf-8") as fixture:
        ltm_dump = yaml.safe_load(fixture)

    node = DummyKnowledgeReuseNode(ltm_dump)
    print(f"Loaded: {args.fixture}")
    print(f"Drives with chains: {list(node.goal_chains)}")
    for drive, chains in node.goal_chains.items():
        print(f"\n{drive}: {len(chains['goals'])} chain entries")
        for goal, domain, policy, pnode, depth in zip(
            chains["goals"],
            chains["domains"],
            chains["policies"],
            chains["pnodes"],
            chains["depth"],
        ):
            print(
                f"  goal={goal!r}, domain={domain!r}, "
                f"policy={policy!r}, pnode={pnode!r}, depth={depth}"
            )

    print(f"\nReuse candidates: {len(node.reuse_candidates)}")
    for candidate in node.reuse_candidates:
        print(f"  {candidate}")


if __name__ == "__main__":
    main()
