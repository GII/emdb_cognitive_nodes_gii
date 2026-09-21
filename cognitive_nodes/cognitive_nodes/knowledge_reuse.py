"""Knowledge-reuse intrinsic motivation and its ROS-free test harness."""

import argparse
from collections import defaultdict
from pathlib import Path

import yaml

try:
    from cognitive_nodes.drive import Drive
    from cognitive_nodes.utils import LTMSubscription
except ModuleNotFoundError as error:
    if error.name != "rclpy":
        raise

    class Drive:
        """Fallback base that keeps the schema harness ROS-free."""

    class LTMSubscription:
        """Fallback mixin that keeps the schema harness ROS-free."""


CHAIN_DEPTH_THRESHOLD = 2


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
        **params,
    ):
        super().__init__(name, class_name, **params)
        if ltm_id is None:
            raise ValueError("No LTM input was provided.")
        if depth_threshold < 0:
            raise ValueError("depth_threshold must be non-negative.")

        self.LTM_id = ltm_id
        self.depth_threshold = depth_threshold
        self.goal_chains = {}
        self.reuse_candidates = []
        self._known_goals = set()
        self.configure_ltm_subscription(ltm_id, self.cbgroup_activation)

    def read_ltm(self, ltm_dump):
        """Refresh the schema and identify newly eligible reuse candidates."""
        previous_goals = self._known_goals
        self.goal_chains = build_goal_chains(ltm_dump)
        current_goals = {
            goal
            for chains in self.goal_chains.values()
            for goal in chains["goals"]
        }
        new_goals = current_goals - previous_goals
        self._known_goals = current_goals
        self.reuse_candidates = self._find_candidates(new_goals)

    def _find_candidates(self, new_goals):
        """Find goal pairs sharing a policy while belonging to other domains."""
        candidates = []
        for drive_name, chains in self.goal_chains.items():
            entries = zip(
                chains["goals"],
                chains["domains"],
                chains["policies"],
                chains["pnodes"],
                chains["depth"],
            )
            entries = list(entries)
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
                        and candidate_depth - depth > self.depth_threshold
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
                                "depth": candidate_depth,
                            }
                        )
        return candidates

    def evaluate(self, perception=None):
        """Activate the drive when at least one reuse candidate is available."""
        self.evaluation.evaluation = float(bool(self.reuse_candidates))
        self.evaluation.timestamp = self.get_clock().now().to_msg()
        return self.evaluation


# Keep the shorter name available for configurations that name drives by their
# intrinsic motivation instead of by their implementation detail.
KnowledgeReuseDrive = DriveKnowledgeReuse


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
        candidates = []
        for drive, chains in self.goal_chains.items():
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
                for (
                    candidate_goal,
                    candidate_domain,
                    candidate_policy,
                    candidate_pnode,
                    candidate_depth,
                ) in entries:
                    if (
                        goal != candidate_goal
                        and policy is not None
                        and policy == candidate_policy
                        and domain != candidate_domain
                        and depth - candidate_depth > self.depth_threshold
                    ):
                        candidates.append(
                            {
                                "drive": drive,
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
