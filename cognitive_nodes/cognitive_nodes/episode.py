from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import xarray as xr

from cognitive_node_interfaces.msg import Episode as EpisodeMsg
from cognitive_processes_interfaces.msg import RewardList
from cognitive_node_interfaces.msg import Action as ActionMsg
from core.utils import perception_dict_to_msg, perception_msg_to_dict, actuation_dict_to_msg, actuation_msg_to_dict



class Container:
    @property
    def name(self) -> str | None:
        return self.data.name

    @name.setter
    def name(self, value: str | None) -> None:
        self.data.name = value

    @property
    def size(self) -> int:
        valid = self._valid_cache
        return int(np.count_nonzero(valid))
    
    @property
    def max_size(self) -> int:
        return int(self.data.sizes["sample"])
    
    @property
    def feature_dim(self) -> str:
        feature_dims = [d for d in self.data.dims if d.endswith("_feature")]
        if len(feature_dims) != 1:
            raise ValueError(f"Container must have exactly one '*_feature' dim, got {feature_dims}")
        return feature_dims[0]
    
    @property
    def feature_labels(self) -> list[str]:
        return self._feature_labels_cache.tolist()

    def __init__(self, name, max_size: int, container_type: str, data_type=np.float64, labels: list = None) -> Container:
        feature_dim = f"{name}_feature"
        feature_labels = labels if labels is not None else []
        shape = (max_size, len(feature_labels))

        coords = {
            feature_dim: feature_labels,
            "timestamp": ("sample", np.zeros(max_size, dtype=np.float64)),
            "buffer_index": ("sample", np.full(max_size, -1).astype(np.uint32)),
            "valid": ("sample", np.zeros(max_size, dtype=bool)),
        }
        attrs = {"type": container_type}
        data = np.full(shape, np.nan, dtype=data_type)

        self.data = xr.DataArray(
            data=data,
            dims=["sample", feature_dim],
            coords=coords,
            attrs=attrs,
            name=name,
        )

        self._feature_index_cache: dict[tuple[tuple[str, ...], tuple[str, ...]], np.ndarray] = {}

        # Ring metadata for O(1) slot resolution.
        self._write_cursor = 0
        self._n_valid = 0

        self._refresh_cached_indices()

    def __len__(self):
        return self.size

    def _refresh_cached_indices(self) -> None:
        self._valid_cache = self.data.coords["valid"].values
        self._buffer_index_cache = self.data.coords["buffer_index"].values
        self._timestamp_cache = self.data.coords["timestamp"].values
        self._feature_cache = self.data.values
        self._feature_labels_cache = self.data.coords[self.feature_dim].values

    def _get_feature_take_indices(
        self,
        src_labels: tuple[str, ...],
        dst_labels: tuple[str, ...],
    ) -> np.ndarray:
        cache_key = (src_labels, dst_labels)
        cached = self._feature_index_cache.get(cache_key)
        if cached is not None:
            return cached

        src_pos = {label: i for i, label in enumerate(src_labels)}
        try:
            take_idx = np.fromiter(
                (src_pos[label] for label in dst_labels),
                dtype=np.int64,
                count=len(dst_labels),
            )
        except KeyError as exc:
            missing = str(exc.args[0])
            raise ValueError(f"Missing destination feature in source container: {missing}") from exc

        self._feature_index_cache[cache_key] = take_idx
        return take_idx
    
    def _is_ring_step_sequence(self, slots: np.ndarray) -> bool:
        if slots.size <= 1:
            return True
        diffs = (slots[1:] - slots[:-1]) % self.max_size
        return bool(np.all(diffs == 1))
    
    def _ordered_all_valid_slots(self) -> np.ndarray:
        n = int(self._n_valid)
        if n == 0:
            return np.empty(0, dtype=np.int64)

        start = (self._write_cursor - n) % self.max_size
        end = start + n

        if end <= self.max_size:
            return np.arange(start, end, dtype=np.int64)

        first = np.arange(start, self.max_size, dtype=np.int64)
        second = np.arange(0, end % self.max_size, dtype=np.int64)
        return np.concatenate((first, second))

    def _resolve_slot(self, index: int, by_buffer_index: bool = False, require_valid: bool = True) -> int:
        if by_buffer_index:
            matches = np.where(self._buffer_index_cache == index)[0]
            if matches.size == 0:
                raise IndexError(f"buffer_index {index} not found")
            slot = int(matches[0])
        else:
            slot = int(index)

        if slot < 0 or slot >= self.max_size:
            raise IndexError(f"slot {slot} out of range [0, {self.max_size - 1}]")

        if require_valid and not bool(self._valid_cache[slot]):
            raise IndexError(f"slot {slot} has no valid data")

        return slot
    
    def _update_buffer_index_fallback(self, slots: np.ndarray) -> np.ndarray:
        # Original robust behavior for non-ring-ordered writes.
        buffer_indexes = self._buffer_index_cache
        valid = self._valid_cache

        old_valid_slots = np.flatnonzero(valid)
        if old_valid_slots.size > 0:
            old_order = np.argsort(buffer_indexes[old_valid_slots], kind="stable")
            old_valid_slots = old_valid_slots[old_order]

        keep_old = old_valid_slots[~np.isin(old_valid_slots, slots)]

        new_order = np.concatenate([keep_old, slots])
        n_valid = new_order.size

        buffer_indexes[:] = -1
        valid[:] = False
        valid[new_order] = True
        buffer_indexes[new_order] = np.arange(n_valid, dtype=np.int64)

        # Reconstruct ring metadata from compact ordering.
        self._n_valid = int(n_valid)
        if self._n_valid == 0:
            self._write_cursor = 0
        else:
            newest_bi = int(self._n_valid - 1)
            newest_slot = int(np.where((valid) & (buffer_indexes == newest_bi))[0][0])
            self._write_cursor = (newest_slot + 1) % self.max_size

        return buffer_indexes[slots]
    
    def _update_buffer_index(self, slots: int | np.ndarray | list[int]) -> np.ndarray:
        """
        Incremental buffer_index update for ring-ordered writes.
        Falls back to robust full rebuild if slots are not ring-contiguous.
        """
        slots = np.asarray(slots, dtype=np.int64).reshape(-1)
        if slots.size == 0:
            return np.empty(0, dtype=np.int64)

        if not self._is_ring_step_sequence(slots):
            return self._update_buffer_index_fallback(slots)

        buffer_indexes = self._buffer_index_cache
        valid = self._valid_cache

        n_new = int(slots.size)
        prev_n = int(self._n_valid)
        cap = self.max_size

        if n_new + prev_n > cap:
            # Wraping case: new slots will overwrite some of the oldest valid slots.
            buffer_indexes = np.remainder(buffer_indexes - n_new, cap)

        else:
            # Non-wrapping case: just mark new slots as valid with correct buffer_index.
            buffer_indexes[slots] = np.arange(prev_n, prev_n + n_new, dtype=np.int64)
        
        valid[slots] = True

        self._n_valid = min(cap, prev_n + n_new)
        self._write_cursor = (self._write_cursor + n_new) % cap

        return buffer_indexes[slots]


    def _resolve_slots_to_write(self, n_samples: int) -> np.ndarray:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        if n_samples > self.max_size:
            raise ValueError(f"n_samples {n_samples} exceeds container size {self.max_size}")

        return (self._write_cursor + np.arange(n_samples, dtype=np.int64)) % self.max_size

    def _write_slot(self, slots: np.ndarray, values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        self._feature_cache[slots] = values
        self._timestamp_cache[slots] = timestamps
        self._valid_cache[slots] = True

        self._update_buffer_index(slots)
        return slots

    def push(
        self,
        sample: Container | np.ndarray,
        timestamps: np.ndarray | None = None,
    ) -> int | np.ndarray:
        if isinstance(sample, Container):
            dst_labels = tuple(str(x) for x in self.feature_labels)
            src_data = sample.data
            src_valid = sample._valid_cache
            if not np.any(src_valid):
                return np.empty(0, dtype=np.int64)

            src_slots = np.flatnonzero(src_valid)
            if src_slots.size > 1:
                src_bi = sample._buffer_index_cache
                src_slots = src_slots[np.argsort(src_bi[src_slots], kind="stable")]

            values = src_data.values[src_slots]
            ts = sample._timestamp_cache[src_slots]
            src_labels = tuple(str(x) for x in sample.feature_labels)

            if src_labels != dst_labels:
                take_idx = self._get_feature_take_indices(src_labels, dst_labels)
                values = np.take(values, take_idx, axis=1)

            values = np.asarray(values, dtype=self.data.dtype)

        else:
            n_features = len(self.feature_labels)
            values = np.asarray(sample, dtype=self.data.dtype)
            if values.ndim == 1:
                values = values.reshape(1, -1)
            elif values.ndim != 2:
                raise ValueError(f"sample ndarray must be 1D or 2D, got {values.shape}")

            if values.shape[1] != n_features:
                raise ValueError(f"sample must have {n_features} features, got {values.shape[1]}")

            if timestamps is None:
                raise ValueError("timestamps ndarray is required when sample is ndarray")
            ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)

        n_rows = values.shape[0]
        if n_rows == 0:
            return np.empty(0, dtype=np.int64)

        if ts.size != n_rows:
            raise ValueError(f"timestamps length must be {n_rows}, got {ts.size}")

        if n_rows > self.max_size:
            values = values[-self.max_size:]
            ts = ts[-self.max_size:]
            n_rows = self.max_size

        slots = self._resolve_slots_to_write(n_rows)
        written_slots = self._write_slot(slots, values, ts)

        return int(written_slots[0]) if written_slots.size == 1 else written_slots

    def clear(self) -> None:
        self._valid_cache[:] = False
        self._buffer_index_cache[:] = -1
        self._timestamp_cache[:] = 0.0

        self._write_cursor = 0
        self._n_valid = 0

    def read(self, index: int | slice | list[int] | np.ndarray | None = None, ordered: bool = True) -> xr.DataArray:
        if index is None:
            if ordered:
                slots = self._ordered_all_valid_slots()
            else:
                slots = np.flatnonzero(self._valid_cache)
            return self.data.isel(sample=slots)

        ordered_slots = self._ordered_all_valid_slots()

        if isinstance(index, int):
            if index < 0 or index >= ordered_slots.size:
                raise IndexError(f"buffer_index {index} out of range [0, {ordered_slots.size - 1}]")
            slots = np.array([ordered_slots[index]], dtype=np.int64)
        elif isinstance(index, slice):
            slots = ordered_slots[index]
        else:
            buffer_indexes = np.asarray(index, dtype=np.int64)
            if buffer_indexes.ndim != 1:
                raise ValueError("index array must be 1-dimensional")
            if np.any((buffer_indexes < 0) | (buffer_indexes >= ordered_slots.size)):
                raise IndexError(f"buffer_index out of range [0, {ordered_slots.size - 1}]")
            slots = ordered_slots[buffer_indexes]

        out = self.data.isel(sample=slots)
        if not ordered or out.sizes.get("sample", 0) <= 1:
            return out

        return out.isel(sample=np.argsort(out.coords["buffer_index"].values, kind="stable"))

    def read_slots(self, slots: int | slice | list[int] | np.ndarray) -> xr.DataArray:
        return self.data.isel(sample=slots)


class Episode:
    """
    Episode class that represents a single episode in the cognitive architecture.
    """
    def __init__(self, old_perception=None, parent_policy='', action=None, perception=None, reward_list=None) -> None:
        """Initialize a new Episode.

        Captures the transition from a previous perceptual state to a new one,
        the selected action, the governing parent policy, and any observed rewards.

        :param old_perception: Perceptual state before the action/transition. If None, an empty dict is used.
        :type old_perception: dict
        :param parent_policy: Identifier of the parent policy responsible for the decision. Defaults to ''.
        :type parent_policy: str
        :param action: The chosen action. If None, a new Action() is created.
        :type action: Action | None
        :param perception: Perceptual state after the action/transition. If None, an empty dict is used.
        :type perception: dict
        :param reward_list: Mapping of goal identifiers to reward values. If None, an empty dict is used.
        :type reward_list: dict[str, float] | None

        :return: None
        :rtype: None

        Notes:
        - old_ltm_state and ltm_state are initialized as empty dicts to store long-term memory snapshots.
        """        


        self.old_perception=old_perception if old_perception is not None else {}
        self.old_ltm_state={}
        self.parent_policy=parent_policy
        self.action=action if action is not None else Action()
        self.perception=perception if perception is not None else {}
        self.ltm_state={}
        self.reward_list=reward_list if reward_list is not None else {}

    def __repr__(self):
        return f"Episode(old_perception={self.old_perception}, parent_policy={self.parent_policy}, action={self.action}, perception={self.perception}, reward_list={self.reward_list})"

class Action:
    """
    Action class used to represent an action in the cognitive architecture.
    """
    def __init__(self, actuation={}, policy_id=None) -> None:
        """Initialize an Action.

        Represents the actuation payload to execute and the policy identifier
        that produced it.

        :param actuation: Mapping of actuator keys to command values. Defaults to {}.
        :type actuation: dict
        :param policy_id: Identifier of the parent policy. If None, it is set to 0.
        :type policy_id: int | None

        :return: None
        :rtype: None

        Notes:
        - policy_id is normalized to int; None becomes 0.
        """        
        self.actuation = actuation
        self.policy_id = policy_id if policy_id is not None else 0

    def __repr__(self):
        return f"Action(actuation={self.actuation}, policy_id={self.policy_id})"


def episode_msg_to_obj(episode_msg: EpisodeMsg) -> Episode:
    """
    Convert a ROS2 Episode message to an Episode object.

    :param episode_msg: The ROS2 Episode message.
    :type episode_msg: cognitive_node_interfaces.msg.Episode
    :return: An Episode object.
    :rtype: Episode
    """
    episode = Episode()
    episode.old_perception = perception_msg_to_dict(episode_msg.old_perception)
    episode.parent_policy = episode_msg.parent_policy
    episode.action = action_msg_to_obj(episode_msg.action)
    episode.perception = perception_msg_to_dict(episode_msg.perception)
    episode.reward_list = reward_msg_to_dict(episode_msg.reward_list)
    return episode

def episode_obj_to_msg(episode: Episode) -> EpisodeMsg:
    """
    Convert an Episode object to a ROS2 Episode message.

    :param episode: The Episode object.
    :type episode: Episode
    :return: A ROS2 Episode message.
    :rtype: cognitive_node_interfaces.msg.Episode
    """
    episode_msg = EpisodeMsg()
    episode_msg.old_perception = perception_dict_to_msg(episode.old_perception)
    episode_msg.parent_policy = episode.parent_policy
    episode_msg.action.actuation = actuation_dict_to_msg(episode.action.actuation)
    episode_msg.action.policy_id = int(episode.action.policy_id)
    episode_msg.perception = perception_dict_to_msg(episode.perception)
    episode_msg.reward_list = reward_dict_to_msg(episode.reward_list)
    return episode_msg

def episode_msg_list_to_obj_list(episode_msg_list: list[EpisodeMsg]) -> list[Episode]:
    """
    Convert a list of ROS2 Episode messages to a list of Episode objects.

    :param episode_msg_list: List of ROS2 Episode messages.
    :type episode_msg_list: list[cognitive_node_interfaces.msg.Episode]
    :return: List of Episode objects.
    :rtype: list[Episode]
    """
    return [episode_msg_to_obj(episode_msg) for episode_msg in episode_msg_list]

def episode_obj_list_to_msg_list(episode_list: list[Episode]) -> list[EpisodeMsg]:
    """
    Convert a list of Episode objects to a list of ROS2 Episode messages.

    :param episode_list: List of Episode objects.
    :type episode_list: list[Episode]
    :return: List of ROS2 Episode messages.
    :rtype: list[cognitive_node_interfaces.msg.Episode]
    """
    return [episode_obj_to_msg(episode) for episode in episode_list]

def action_msg_to_obj(action_msg) -> Action:
    """
    Convert a ROS2 action message to an Action object.

    :param action_msg: The ROS2 action message.
    :type action_msg: cognitive_node_interfaces.msg.Action
    :return: An Action object.
    :rtype: Action
    """
    action = Action()
    action.actuation = actuation_msg_to_dict(action_msg.actuation)
    action.policy_id = action_msg.policy_id
    return action

def action_obj_to_msg(action: Action):
    """
    Convert an Action object to a ROS2 action message.

    :param action: The Action object.
    :type action: Action
    :return: A ROS2 action message.
    :rtype: cognitive_node_interfaces.msg.Action
    """
    action_msg = ActionMsg()
    action_msg.actuation = actuation_dict_to_msg(action.actuation)
    action_msg.policy_id = action.policy_id
    return action_msg

def action_msg_list_to_obj_list(action_msg_list: list[ActionMsg]) -> list[Action]:
    """
    Convert a list of ROS2 action messages to a list of Action objects.

    :param action_msg_list: List of ROS2 action messages.
    :type action_msg_list: list[cognitive_node_interfaces.msg.Action]
    :return: List of Action objects.
    :rtype: list[Action]
    """
    return [action_msg_to_obj(action_msg) for action_msg in action_msg_list]

def action_obj_list_to_msg_list(action_list: list[Action]) -> list[ActionMsg]:
    """
    Convert a list of Action objects to a list of ROS2 action messages.

    :param action_list: List of Action objects.
    :type action_list: list[Action]
    :return: List of ROS2 action messages.
    :rtype: list[cognitive_node_interfaces.msg.Action]
    """
    return [action_obj_to_msg(action) for action in action_list]

def reward_dict_to_msg(reward_dict):
    """
    Convert a reward dictionary to a ROS2 message format.

    :param reward_dict: The reward dictionary.
    :type reward_dict: dict
    :return: A ROS2 message representing the reward.
    :rtype: cognitive_node_interfaces.msg.Reward
    """
    reward_msg = RewardList()
    reward_msg.goals = list(reward_dict.keys())
    reward_msg.goals = [str(goal) for goal in reward_msg.goals]
    reward_msg.rewards = list(reward_dict.values())
    reward_msg.rewards = [float(reward) for reward in reward_msg.rewards]
    return reward_msg

def reward_msg_to_dict(reward_msg: RewardList) -> dict:
    """
    Convert a ROS2 reward message to a dictionary.

    :param reward_msg: The ROS2 reward message.
    :type reward_msg: cognitive_node_interfaces.msg.RewardList
    :return: A dictionary representing the rewards.
    :rtype: dict
    """
    return {goal: reward for goal, reward in zip(reward_msg.goals, reward_msg.rewards)}