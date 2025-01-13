import yaml

from std_msgs.msg import String
from cognitive_node_interfaces.msg import SuccessRate
from core.utils import class_from_classname


class LTMSubscription:
    def configure_ltm_subscription(self, ltm):
        self.ltm_suscription = self.create_subscription(String, "state", self.ltm_change_callback, 0, callback_group=self.cbgroup_client)
        
    def ltm_change_callback(self, msg):
        self.get_logger().info("Processing change from LTM...")
        ltm_dump = yaml.safe_load(msg.data)
        self.read_ltm(ltm_dump=ltm_dump)

    def read_ltm(self, ltm_dump):
        raise NotImplementedError
    
class PNodeSuccess(LTMSubscription):
    def configure_pnode_success(self, ltm):
        self.configure_ltm_subscription(ltm)
        self.pnode_subscriptions = {}
        self.pnode_evaluation={}

    def read_ltm(self, ltm_dump):
        pnodes = ltm_dump["PNode"]
        for pnode in pnodes:
            if pnode not in self.pnode_subscriptions.keys():
                self.pnode_subscriptions[pnode] = self.create_subscription(SuccessRate, f"/pnode/{pnode}/success_rate", self.pnode_success_callback, 1, callback_group=self.cbgroup_activation)

    def pnode_success_callback(self, msg: SuccessRate):
        pnode = msg.node_name
        goal_linked = msg.flag
        success_rate = msg.success_rate
        self.pnode_evaluation[pnode] = success_rate * (not goal_linked)

class EpisodeSubscription:
    def configure_episode_subscription(self, episode_topic, episode_msg):
        msg_obj=class_from_classname(episode_msg)
        self.ltm_suscription = self.create_subscription(msg_obj, episode_topic, self.episode_callback, 0, callback_group=self.cbgroup_activation)
    
    def episode_callback(self, msg):
        raise NotImplementedError

  