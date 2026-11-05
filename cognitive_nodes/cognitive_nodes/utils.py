import yaml
import inspect

from std_msgs.msg import String
from cognitive_node_interfaces.msg import SuccessRate
from core.utils import class_from_classname


class LTMSubscription:
    """
    LTMSubscription is a mixin class that provides a method to configure a subscription to the LTM.
    """    
    def configure_ltm_subscription(self, ltm, callback_group):
        """
        Configure the subscription to the LTM state topic (``state``).

        :param ltm: LTM ID. Not used.
        :type ltm: str
        :param callback_group: Callback group of the subscription.
        :type callback_group: rclpy.callback_groups.CallbackGroup
        """        
        self.ltm_subscription = self.create_subscription(String, "state", self.ltm_change_callback, 0, callback_group=callback_group)

    async def ltm_change_callback(self, msg):
        """
        Callback that processes the LTM message. It parses the YAML in the message and calls ``read_ltm`` (awaiting it if it is a coroutine).
        This is a coroutine and must be awaited if called directly.

        :param msg: Message from the LTM.
        :type msg: std_msgs.msg.String
        """        
        self.get_logger().info("Processing change from LTM...")
        ltm_dump = yaml.safe_load(msg.data)

        if inspect.iscoroutinefunction(self.read_ltm):
            await self.read_ltm(ltm_dump=ltm_dump)
        else:
            self.read_ltm(ltm_dump=ltm_dump)

    async def read_ltm(self, ltm_dump):
        """
        Placeholder for LTM processing, to be implemented in the class that uses the mixin (it may be a plain method or a coroutine).

        :param ltm_dump: Dictionary with the data from the LTM
        :type ltm_dump: dict
        :raises NotImplementedError: Method must be implemented in the subclass.
        """        
        raise NotImplementedError
    
class PNodeSuccess(LTMSubscription):
    """
    PNodeSuccess is a mixin class that provides a method to configure a subscription to the success rate of the P-Nodes.
    """    
    def configure_pnode_success(self, ltm, callback_group):
        """
        Configure the subscription to the success rate of the P-Nodes. It subscribes to the LTM (see :meth:`LTMSubscription.configure_ltm_subscription`)
        and creates the dictionaries ``pnode_subscriptions`` and ``pnode_evaluation``.

        :param ltm: LTM id.
        :type ltm: str
        :param callback_group: Callback group of the LTM subscription.
        :type callback_group: rclpy.callback_groups.CallbackGroup
        """        
        self.configure_ltm_subscription(ltm, callback_group)
        self.pnode_subscriptions = {}
        self.pnode_evaluation={}

    def read_ltm(self, ltm_dump):
        """
        Method that processes the LTM data and subscribes to the success rate (``/pnode/<name>/success_rate``) of the P-Nodes that were not subscribed yet.
        Subscriptions of P-Nodes that disappear from the LTM are not removed.

        :param ltm_dump: Dictionary with the data from the LTM. It must have a ``PNode`` key.
        :type ltm_dump: dict
        """        
        pnodes = ltm_dump["PNode"]
        for pnode in pnodes:
            if pnode not in self.pnode_subscriptions.keys():
                self.pnode_subscriptions[pnode] = self.create_subscription(SuccessRate, f"/pnode/{pnode}/success_rate", self.pnode_success_callback, 1, callback_group=self.cbgroup_activation)

    def pnode_success_callback(self, msg: SuccessRate):
        """
        Callback that processes the success rate of a P-Node. It stores in ``pnode_evaluation`` the success rate, or 0.0 if the P-Node is already linked to a goal (``flag``).

        :param msg: Success rate message.
        :type msg: cognitive_node_interfaces.msg.SuccessRate
        """        
        pnode = msg.node_name
        goal_linked = msg.flag
        success_rate = msg.success_rate
        self.pnode_evaluation[pnode] = success_rate * (not goal_linked)

class EpisodeSubscription:
    """
    EpisodeSubscription is a mixin class that provides a method to configure a subscription to the episodes.
    """    
    def configure_episode_subscription(self, episode_topic, episode_msg, callback_group):
        """
        Configure the subscription to the episodes. Messages are passed to ``episode_callback``.

        :param episode_topic: Name of the topic where the episodes are published.
        :type episode_topic: str
        :param episode_msg: Class name of the message type of the episodes (resolved with :func:`core.utils.class_from_classname`).
        :type episode_msg: str
        :param callback_group: Callback group of the subscription.
        :type callback_group: rclpy.callback_groups.CallbackGroup
        """        
        msg_obj=class_from_classname(episode_msg)
        self.episode_subscription = self.create_subscription(msg_obj, episode_topic, self.episode_callback, 0, callback_group=callback_group)

    def episode_callback(self, msg):
        """
        Callback that processes the episodes.

        :param msg: Message of the episodes topic, containing a flattened episode.
        :type msg: ROS2 message. Typically core_interfaces.msg.Container
        :raises NotImplementedError: Method must be implemented in the subclass.
        """        
        raise NotImplementedError

  