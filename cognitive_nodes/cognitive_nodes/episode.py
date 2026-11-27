from __future__ import annotations
import numpy as np
import xarray as xr
from copy import copy
from rclpy.time import Time

from core.container import Container, consolidate_containers

from core_interfaces.msg import Container as ContainerMsg




class Episode:
    """
    Episode class that represents one or more consecutive transitions of the cognitive architecture: the perception
    before an action, the action, the perception after it and the rewards obtained.

    An episode is made of four optional :class:`core.container.Container` objects, all of them with the same number of rows
    (one row per transition):

    - ``old_perception``: perception before the action (container type ``"perception"``).
    - ``action``: the action (container type ``"action"``).
    - ``perception``: perception after the action (container type ``"perception"``).
    - ``rewards``: reward obtained for each goal (container type ``"rewards"``). Its columns are labeled
      ``rewards:<goal>``, and a goal without a reward in a row is NaN.

    The first container assigned fixes the number of rows (``container_size``, also returned by ``len(episode)``); any
    other container with a different number of rows is rejected with a ``ValueError``. The setters of the four fields
    store a shallow copy of the Container (``copy.copy``) and set its ``name`` to the field name and its ``container_type``
    as listed above. The name matters because the labels of the flattened episode (see :meth:`obtain_flattened_episode`)
    are ``<name>:<feature label>``. The underlying data array is not duplicated by the copy, so it is shared with the
    Container that was passed in. Assigning ``None`` to a field leaves it as ``None``, not as an empty Container or dict.

    The episode also stores the identifier of the parent policy, the domain name, and the optional snapshots of the long-term
    memory before and after the transition (``old_ltm_state`` and ``ltm_state``), which are not part of the flattened episode.
    """
    def __init__(self, old_perception: Container = None, parent_policy='', domain_name='', action: Container = None, perception: Container = None, rewards: Container = None, old_ltm_state: dict = {}, ltm_state: dict = {}) -> None:
        """Initialize a new Episode.

        Captures the transition from a previous perceptual state to a new one,
        the selected action, the governing parent policy, and any observed rewards.
        Each Container that is not None is assigned through its property, so it is copied, renamed and checked
        against the number of rows of the others (see the class description).

        :param old_perception: Perceptual state before the action/transition. If None, the field stays None.
        :type old_perception: Container
        :param parent_policy: Identifier of the parent policy responsible for the decision. Defaults to ''.
        :type parent_policy: str
        :param domain_name: Name of the domain the episode belongs to. Defaults to ''.
        :type domain_name: str
        :param action: The chosen action. If None, the field stays None.
        :type action: Container
        :param perception: Perceptual state after the action/transition. If None, the field stays None.
        :type perception: Container
        :param rewards: Container with one column ``rewards:<goal>`` per goal. If None, the field stays None.
        :type rewards: Container
        :param old_ltm_state: Snapshot of long-term memory state before the transition. Defaults to an empty dict; None is also replaced by an empty dict.
        :type old_ltm_state: dict
        :param ltm_state: Snapshot of long-term memory state after the transition. Defaults to an empty dict; None is also replaced by an empty dict.
        :type ltm_state: dict

        :return: None
        :rtype: None
        :raises ValueError: If the Containers don't all have the same number of rows.
        """        

        self.container_size: int | None = None
        self._old_perception: Container = None
        self._action: Container = None
        self._perception: Container = None
        self._rewards: Container = None

        self.old_ltm_state = {} if old_ltm_state is None else old_ltm_state
        self.ltm_state = {} if ltm_state is None else ltm_state
        self.parent_policy = parent_policy
        self.domain_name = domain_name

        self.field_type_map = {
            "old_perception": "perception",
            "action": "action",
            "perception": "perception",
            "rewards": "rewards",
        }

        if old_perception is not None:
            self.old_perception = old_perception
        if action is not None:
            self.action = action
        if perception is not None:
            self.perception = perception
        if rewards is not None:
            self.rewards = rewards

    def __len__(self):
        """Number of rows (transitions) of the episode.

        :return: The number of rows shared by all the Containers, or 0 if none has been assigned yet.
        :rtype: int
        """
        return self.container_size if self.container_size is not None else 0


    def _bind_container(self, field_name: str, container: Container) -> None:
        """Store a shallow copy of a Container in one of the fields of the episode.

        The copy is renamed to the field name and its container_type is taken from field_type_map. The first Container
        stored fixes container_size. Passing None sets the field to None.

        :param field_name: Name of the field: "old_perception", "action", "perception" or "rewards".
        :type field_name: str
        :param container: The Container to store, or None.
        :type container: Container | None
        :raises ValueError: If the field name is not an attribute of the episode, or the number of rows of the Container differs from container_size.
        """
        if not hasattr(self, field_name):
            raise ValueError(f"Invalid field name: {field_name}")
        container = copy(container)  # Avoid modifying the original container passed in

        if container is None:
            setattr(self, f"_{field_name}", None)
            return

        size = len(container)
        if self.container_size is None:
            self.container_size = size
        elif size != self.container_size:
            raise ValueError(
                f"{field_name} size {size} does not match episode size {self.container_size}"
            )
        container.name = field_name
        container.container_type = self.field_type_map.get(field_name, field_name)
        setattr(self, f"_{field_name}", container)

    @property
    def old_perception(self) -> Container | None:
        """Perception before the action. Setting it stores a renamed shallow copy (name "old_perception", type "perception"); None leaves it as None.

        :raises ValueError: When setting a Container whose number of rows differs from that of the episode.
        """
        return self._old_perception

    @old_perception.setter
    def old_perception(self, container: Container) -> None:
        self._bind_container("old_perception", container)

    @property
    def action(self) -> Container | None:
        """The action. Setting it stores a renamed shallow copy (name "action", type "action"); None leaves it as None.

        :raises ValueError: When setting a Container whose number of rows differs from that of the episode.
        """
        return self._action

    @action.setter
    def action(self, container: Container) -> None:
        self._bind_container("action", container)

    @property
    def perception(self) -> Container | None:
        """Perception after the action. Setting it stores a renamed shallow copy (name "perception", type "perception"); None leaves it as None.

        :raises ValueError: When setting a Container whose number of rows differs from that of the episode.
        """
        return self._perception

    @perception.setter
    def perception(self, container: Container) -> None:
        self._bind_container("perception", container)

    @property
    def rewards(self) -> Container | None:
        """Rewards, with a column ``rewards:<goal>`` per goal. Setting it stores a renamed shallow copy (name "rewards", type "rewards"); None leaves it as None.

        :raises ValueError: When setting a Container whose number of rows differs from that of the episode.
        """
        return self._rewards

    @rewards.setter
    def rewards(self, container: Container) -> None:
        self._bind_container("rewards", container)

    @property
    def reward_list(self) -> dict|list[dict]|None:
        """Rewards of the episode as dictionaries from goal identifier to reward value.

        The goal identifier is the part of each column label after the first ``:`` (``rewards:<goal>``). Goals whose value
        is NaN in a row are left out.

        :return: A list with one dict per row if the episode has more than one row; a single dict if it has exactly one row;
            an empty dict if there are no rewards (rewards is None or has no rows).
        :rtype: dict | list[dict]
        """
        if self.rewards is None:
            return {}
        reward_dicts = []
        data = self.rewards.read().values
        for row in data:
            reward_dict = {
                label.split(":", 1)[1]: value
                for label, value in zip(self.rewards.feature_labels, row)
                if not np.isnan(value)
            }
            reward_dicts.append(reward_dict)
        return reward_dicts if len(reward_dicts) > 1 else reward_dicts[0] if reward_dicts else {}

    def update_reward(self, reward_dict_list: dict | list[dict], timestamp: float|Time) -> None:
        """Update the episode's rewards with new reward values.

        :param reward_dict_list: A dictionary (or list of dictionaries) mapping goal identifiers to reward values.
        :type reward_dict_list: dict | list[dict]
        :param timestamp: A single timestamp for all the rewards pushed. A ``rclpy.time.Time`` is converted to seconds (nanoseconds / 1e9); a float is used as is.
        :type timestamp: float | Time
        :return: None
        :rtype: None
        :raises ValueError: If the resulting rewards Container does not have the same number of rows as the rest of the episode.

        Notes:

        - Each dictionary is pushed as one row of the ``rewards`` Container, with columns ``rewards:<goal>``. If the episode has no rewards yet, the Container is created with ``max_size=container_size``; columns for new goals are added to an existing one, and goals missing in a row are filled with NaN.
        - The Container is assigned through the ``rewards`` setter, so it is copied and its size is checked.
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
                rewards_container = Container("rewards", max_size=self.container_size, container_type="rewards", labels=labels)
                rewards = np.fromiter((reward_dict[g] for g in goals), dtype=rewards_container.data.dtype)
            else:
                rewards = np.fromiter((reward_dict.get(goal) for goal in goals), dtype=rewards_container.data.dtype)
            rewards_container.push(rewards, src_labels=labels, timestamps=timestamp, extend_labels=True, allow_missing=True, fill_value=np.nan)
        self.rewards = rewards_container

    def obtain_flattened_episode(self) -> Container:
        """Join the populated Containers of the episode into a single Container.

        The parts, in order, are old_perception, action, perception and rewards; those that are None are skipped. The columns
        of the result are labeled ``<field name>:<feature label>`` (see :func:`core.container.consolidate_containers`), the
        result has name and type ``"episode"``, and its attributes ``parent_policy`` and ``domain_name`` are those of the episode.
        The LTM snapshots are not included.

        :return: The flattened episode, or None if all the fields are None.
        :rtype: Container | None
        """
        parts = [
            self.old_perception,
            self.action,
            self.perception,
            self.rewards,
        ]
        populated_parts = [p for p in parts if p is not None]
        if not populated_parts:
            return None
        consolidated = consolidate_containers(populated_parts, name="episode", container_type="episode", attrs={"parent_policy": self.parent_policy, "domain_name": self.domain_name})
        return consolidated

    def __repr__(self):
        return f"Episode(old_perception={self.old_perception}, parent_policy={self.parent_policy}, domain_name={self.domain_name}, action={self.action}, perception={self.perception}, rewards={self.rewards})"


def container_msg_to_episode(msg: ContainerMsg) -> Episode:
    """Convert a flattened ContainerMsg back into a structured Episode (split by 'prefix:feature').

    The message is decoded with ``Container.from_msg`` and then converted with :func:`container_to_episode_obj`.

    :param msg: Message with a flattened episode, as produced by :func:`episode_obj_to_msg`.
    :type msg: core_interfaces.msg.Container
    :return: The episode.
    :rtype: Episode
    """
    cont = Container.from_msg(msg)
    return container_to_episode_obj(cont)


def container_to_episode_obj(container: Container|xr.DataArray) -> Episode:
    """Convert a flattened Container back into a structured Episode (split by 'prefix:feature').

    The columns are grouped by the prefix of their label (the text before the first ``:``). Each group becomes a Container
    named and typed after the prefix, with the prefix removed from its labels, and it is assigned to the field of the same name
    (old_perception, action, perception or rewards). Columns without ``:`` are grouped under the prefix "unknown", and groups
    whose prefix is not one of the four fields are dropped. ``parent_policy`` and ``domain_name`` are read from the attributes
    of the data (empty strings if absent).

    :param container: A flattened episode, or its DataArray.
    :type container: Container | xarray.DataArray
    :return: The episode. An empty Episode if the input is None or an empty Container.
    :rtype: Episode
    :raises ValueError: If the input is neither a Container, an xarray.DataArray nor None.
    """
    epi = Episode()

    if isinstance(container, Container):
        if container.size == 0:
            return epi
        epi.parent_policy = container.data.attrs.get("parent_policy", "")
        epi.domain_name = container.data.attrs.get("domain_name", "")
        data = container.read()
    elif isinstance(container, xr.DataArray):
        data = container
        epi.parent_policy = data.attrs.get("parent_policy", "")
        epi.domain_name = data.attrs.get("domain_name", "")
    elif container is None:
        return epi
    else:
        raise ValueError(f"Input must be a Container or xarray DataArray not {type(container)}")


    values = data.values
    ts = data.coords["timestamp"].values
    feature_labels = data.coords["features"].values
    dtype = data.dtype
    attrs = data.attrs
    attrs.pop("parent_policy", None)  # Remove parent_policy from attrs since it's now stored in the episode object
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

    return epi

def episode_obj_to_msg(episode: Episode, name="episodes") -> ContainerMsg:
    """
    Convert an Episode object to a ROS2 Container message.

    :param episode: The Episode object.
    :type episode: Episode
    :param name: Not used.
    :type name: str
    :return: A ROS2 Container message with the flattened episode (see :meth:`Episode.obtain_flattened_episode`), or an empty message if the episode has no populated field.
    :rtype: core_interfaces.msg.Container
    """
    cont = episode.obtain_flattened_episode()
    return cont.to_msg() if cont is not None else ContainerMsg()