import yaml

from std_msgs.msg import String
from cognitive_node_interfaces.msg import SuccessRate
from core.utils import class_from_classname


class LTMSubscription:
    """
    LTMSubscription is a mixin class that provides a method to configure a subscription to the LTM.
    """    
    def configure_ltm_subscription(self, ltm):
        """
        Configure the subscription to the LTM.

        :param ltm: LTM ID.
        :type ltm: str
        """        
        self.ltm_suscription = self.create_subscription(String, "state", self.ltm_change_callback, 0, callback_group=self.cbgroup_client)
        
    def ltm_change_callback(self, msg):
        """
        Callback that processes the LTM message.

        :param msg: Message from the LTM.
        :type msg: std_msgs.msg.String
        """        
        self.get_logger().info("Processing change from LTM...")
        ltm_dump = yaml.safe_load(msg.data)
        self.read_ltm(ltm_dump=ltm_dump)

    def read_ltm(self, ltm_dump):
        """
        Placeholder for LTM processing.

        :param ltm_dump: Dictionary with the data from the LTM
        :type ltm_dump: dict
        :raises NotImplementedError: Method must be implemented in the subclass.
        """        
        raise NotImplementedError
    
class PNodeSuccess(LTMSubscription):
    """
    PNodeSuccess is a mixin class that provides a method to configure a subscription to the success rate of the P-Nodes.
    """    
    def configure_pnode_success(self, ltm):
        """
        Configure the subscription to the success rate of the P-Nodes.

        :param ltm: LTM id.
        :type ltm: str
        """        
        self.configure_ltm_subscription(ltm)
        self.pnode_subscriptions = {}
        self.pnode_evaluation={}

    def read_ltm(self, ltm_dump):
        """
        Method that processes the LTM data and subscribes to the success rate of the P-Nodes.

        :param ltm_dump: Dictionary with the data from the LTM.
        :type ltm_dump: str
        """        
        pnodes = ltm_dump["PNode"]
        for pnode in pnodes:
            if pnode not in self.pnode_subscriptions.keys():
                self.pnode_subscriptions[pnode] = self.create_subscription(SuccessRate, f"/pnode/{pnode}/success_rate", self.pnode_success_callback, 1, callback_group=self.cbgroup_activation)

    def pnode_success_callback(self, msg: SuccessRate):
        """
        Callback that processes the success rate of a P-Node.

        :param msg: Success rate message.
        :type msg: SuccessRate
        """        
        pnode = msg.node_name
        goal_linked = msg.flag
        success_rate = msg.success_rate
        self.pnode_evaluation[pnode] = success_rate * (not goal_linked)

class EpisodeSubscription:
    """
    EpisodeSubscription is a mixin class that provides a method to configure a subscription to the episodes.
    """    
    def configure_episode_subscription(self, episode_topic, episode_msg):
        """
        Configure the subscription to the episodes.

        :param episode_topic: Name of the topic where the episodes are published.
        :type episode_topic: str
        :param episode_msg: Message type of the episodes.
        :type episode_msg: str
        """        
        msg_obj=class_from_classname(episode_msg)
        self.ltm_suscription = self.create_subscription(msg_obj, episode_topic, self.episode_callback, 0, callback_group=self.cbgroup_activation)
    
    def episode_callback(self, msg):
        """
        Callback that processes the episodes.

        :param msg: Episode message.
        :type msg: ROS2 message. Typically cognitive_process_interfaces.msg.Episode
        :raises NotImplementedError: Method must be implemented in the subclass.
        """        
        raise NotImplementedError

  