from __future__ import annotations
import numpy as np
from rclpy.time import Time

from core.container import Container, consolidate_containers

from core_interfaces.msg import Container as ContainerMsg




class Episode:
    """
    Episode class that represents a single episode in the cognitive architecture.
    """
    def __init__(self, old_perception: Container = None, parent_policy='', action: Container = None, perception: Container = None, rewards: Container = None, old_ltm_state: dict = {}, ltm_state: dict = {}) -> None:
        """Initialize a new Episode.

        Captures the transition from a previous perceptual state to a new one,
        the selected action, the governing parent policy, and any observed rewards.

        :param old_perception: Perceptual state before the action/transition. If None, an empty dict is used.
        :type old_perception: Container
        :param parent_policy: Identifier of the parent policy responsible for the decision. Defaults to ''.
        :type parent_policy: str
        :param action: The chosen action.
        :type action: Container
        :param perception: Perceptual state after the action/transition. If None, an empty dict is used.
        :type perception: Container
        :param rewards: Mapping of goal identifiers to reward values. If None, an empty dict is used.
        :type rewards: Container
        :param old_ltm_state: Snapshot of long-term memory state before the transition. Defaults to an empty dict.
        :type old_ltm_state: dict
        :param ltm_state: Snapshot of long-term memory state after the transition. Defaults to an empty dict.
        :type ltm_state: dict

        :return: None
        :rtype: None

        Notes:
        - old_ltm_state and ltm_state are initialized as empty dicts to store long-term memory snapshots.
        """        

        self.container_size: int | None = None
        self._old_perception: Container = None
        self._action: Container = None
        self._perception: Container = None
        self._rewards: Container = None

        self.old_ltm_state = {} if old_ltm_state is None else old_ltm_state
        self.ltm_state = {} if ltm_state is None else ltm_state
        self.parent_policy = parent_policy

        if old_perception is not None:
            self.old_perception = old_perception
        if action is not None:
            self.action = action
        if perception is not None:
            self.perception = perception
        if rewards is not None:
            self.rewards = rewards


    def _bind_container(self, field_name: str, container: Container) -> None:
        if container is None:
            raise ValueError(f"{field_name} cannot be None")

        size = len(container)
        if self.container_size is None:
            self.container_size = size
        elif size != self.container_size:
            raise ValueError(
                f"{field_name} size {size} does not match episode size {self.container_size}"
            )

        setattr(self, f"_{field_name}", container)

    @property
    def old_perception(self) -> Container | None:
        return self._old_perception

    @old_perception.setter
    def old_perception(self, container: Container) -> None:
        self._bind_container("old_perception", container)

    @property
    def action(self) -> Container | None:
        return self._action

    @action.setter
    def action(self, container: Container) -> None:
        self._bind_container("action", container)

    @property
    def perception(self) -> Container | None:
        return self._perception

    @perception.setter
    def perception(self, container: Container) -> None:
        self._bind_container("perception", container)

    @property
    def rewards(self) -> Container | None:
        return self._rewards

    @rewards.setter
    def rewards(self, container: Container) -> None:
        self._bind_container("rewards", container)

    @property
    def reward_list(self) -> dict|list[dict]|None:
        if self.rewards is None:
            return None
        reward_dicts = []
        data = self.rewards.read().values
        for row in data:
            reward_dict = {label.split(":")[1]: row[idx] for idx, label in enumerate(self.rewards.feature_labels)}
            reward_dicts.append(reward_dict)
        return reward_dicts if len(reward_dicts) > 1 else reward_dicts[0] if reward_dicts else None

    def update_reward(self, reward_dict_list: dict | list[dict], timestamp: float|Time) -> None:
        """Update the episode's rewards with new reward values.

        :param reward_dict: A dictionary mapping goal identifiers to reward values.
        :type reward_dict: dict
        :param timestamp: A single timestamp for the rewards.
        :type timestamp: float | Time
        :return: None
        :rtype: None
        """
        timestamp = timestamp.nanoseconds / 1e9 if isinstance(timestamp, Time) else timestamp
        if isinstance(reward_dict_list, dict):
            reward_dict_list = [reward_dict_list]
        
        rewards_container = self.rewards
        for reward_dict in reward_dict_list:
            goals = list(reward_dict.keys())
            labels = [f"rewards:{goal}" for goal in goals]
            if rewards_container is None:
                # If rewards container doesn't exist, create it with the goals as labels
                rewards_container = Container("rewards", max_size=self.container_size, container_type="dict", labels=labels)
                rewards = np.fromiter((reward_dict[g] for g in goals), dtype=rewards_container.data_type)
            else:
                # If rewards container exists, update it with new goals and rewards, rewards not present in the reward_dict will be set to 0.0.
                # If new goals are introduced, a new rewards container will be created with the updated set of goals as labels.
                existing_goals = set([label.split(":")[1] for label in rewards_container.labels])
                new_goals = set(goals)
                all_goals = existing_goals.union(new_goals)
                rewards = np.fromiter((reward_dict.get(goal, 0.0) for goal in all_goals), dtype=rewards_container.data_type)
                if not new_goals.issubset(existing_goals):
                    all_goals = list(existing_goals.union(new_goals))
                    if self.container_size > 1:
                        # TODO: Implement handling new goals in rewards update for container_size > 1
                        # Requires reconstructing the rewards container with the new set of goals and properly aligning existing reward values with the new labels, filling in 0.0 for any missing rewards in existing entries. 
                        raise NotImplementedError("Handling new goals in rewards update is only implemented for container_size=1")
                    rewards_container = Container("rewards", max_size=self.container_size, container_type="dict", labels=all_goals)
            rewards_container.push(rewards, labels=labels, timestamp=timestamp)
        self.rewards = rewards_container

    def obtain_flattened_episode(self) -> Container:
        parts = [
            self.old_perception,
            self.action,
            self.perception,
            self.rewards,
        ]
        populated_parts = [p for p in parts if p is not None]
        if not populated_parts:
            return None
        consolidated = consolidate_containers(populated_parts, container_type="episode", attrs={"parent_policy": self.parent_policy})
        return consolidated

    def __repr__(self):
        return f"Episode(old_perception={self.old_perception}, parent_policy={self.parent_policy}, action={self.action}, perception={self.perception}, rewards={self.rewards})"


def container_msg_to_episode(msg: ContainerMsg) -> Episode:
    """Convert a flattened ContainerMsg back into a structured Episode (split by 'prefix:feature')."""
    cont = Container.from_msg(msg)
    return container_to_episode_obj(cont)


def container_to_episode_obj(container: Container) -> Episode:
    """Convert a flattened Container back into a structured Episode (split by 'prefix:feature')."""
    epi = Episode()
    if container.size == 0:
        return epi

    epi.parent_policy = container.attrs.get("parent_policy", "")
    data = container.read()
    values = data.values
    ts = data.coords["timestamp"].values
    feature_labels = data.coords["feature"].values
    dtype = data.dtype
    attrs = data.attrs
    n_rows = int(values.shape[0])

    # set episode container size before assigning parts so setters validate consistently
    epi.container_size = n_rows

    # group columns by prefix (prefix is the part name before the first ':')
    groups: dict[str, list[tuple[int, str]]] = {}
    order: list[str] = []
    for col_idx, full_label in enumerate(feature_labels):
        if ":" in full_label:
            prefix, lbl = full_label.split(":", 1)
        else:
            prefix, lbl = "unknown", full_label
        if prefix not in groups:
            groups[prefix] = []
            order.append(prefix)
        groups[prefix].append((col_idx, lbl))

    # create a Container for each group and bind to episode
    for prefix in order:
        cols = groups[prefix]
        col_indices = [c[0] for c in cols]
        sub_labels = [c[1] for c in cols]
        sub_values = values[:, col_indices] if n_rows > 0 else np.empty((0, len(sub_labels)), dtype=dtype)
        sub_ts = ts if ts.size > 0 else np.empty(0, dtype=np.float64)

        c = Container(name=prefix, max_size=n_rows, container_type=prefix, data_type=dtype, labels=sub_labels, attrs=attrs)
        c.push(sub_values, src_labels=sub_labels, src_dtype=dtype, timestamps=sub_ts)
        # assign to episode (will validate sizes)
        if prefix == "old_perception":
            epi.old_perception = c
        elif prefix == "action":
            epi.action = c
        elif prefix == "perception":
            epi.perception = c
        elif prefix == "rewards":
            epi.rewards = c
        else:
            # attach unknown prefixes as attributes for future use
            setattr(epi, prefix, c)

    return epi

def episode_obj_to_msg(episode: Episode, name="episodes") -> ContainerMsg:
    """
    Convert an Episode object to a ROS2 Episode message.

    :param episode: The Episode object.
    :type episode: Episode
    :return: A ROS2 Episode message.
    :rtype: cognitive_node_interfaces.msg.Episode
    """
    cont = episode.obtain_flattened_episode()
    return cont.to_msg() if cont is not None else ContainerMsg()