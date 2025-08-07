from cognitive_node_interfaces.msg import Episode as EpisodeMsg
from cognitive_processes_interfaces.msg import RewardList
from cognitive_node_interfaces.msg import Action as ActionMsg
from core.utils import perception_dict_to_msg, perception_msg_to_dict, actuation_dict_to_msg, actuation_msg_to_dict


class Episode:
    """
    Episode class used as STM (Short Term Memory) for the cognitive architecture.
    """
    def __init__(self, old_perception=None, parent_policy='', action=None, perception=None, reward_list=None) -> None:


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