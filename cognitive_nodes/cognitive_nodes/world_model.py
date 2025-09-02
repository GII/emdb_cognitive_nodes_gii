import rclpy
from copy import deepcopy

from cognitive_nodes.generic_model import GenericModel, Learner
from simulators.scenarios_2D import SimpleScenario, EntityType
from cognitive_node_interfaces.msg import Perception, Actuation
from core.utils import actuation_dict_to_msg, actuation_msg_to_dict, perception_dict_to_msg, perception_msg_to_dict
from rclpy.impl.rcutils_logger import RcutilsLogger
from cognitive_node_interfaces.msg import Perception, PerceptionStamped, SuccessRate
from rclpy.time import Time



class WorldModel(GenericModel):
    """
    World Model class: A static world model that is always active
    """
    def __init__(self, name='world_model', class_name = 'cognitive_nodes.world_model.WorldModel', **params):
        """
        Constructor of the World Model class.

        Initializes a World Model instance with the given name and registers it in the LTM.

        :param name: The name of the World Model instance.
        :type name: str
        :param class_name: The name of the World Model class.
        :type class_name: str
        """
        super().__init__(name, class_name, node_type="world_model", **params)
        self.episodic_buffer=None
        self.learner=None
        self.confidence_evaluator=None
        self.activation.activation = 1.0
    
    def predict(self, perception, action):
        raise NotImplementedError


class Sim2DWorldModel(WorldModel):
    """
    Sim2DWorldModel class: A fixed world model of a 2D simulator. It uses the SimpleScenario simulator to predict the next perception.
    """    
    def __init__(self, name='world_model', actuation_config=None, perception_config=None, class_name='cognitive_nodes.world_model.WorldModel', **params):
        """
        Constructor of the Sim2DWorldModel class.

        :param name: The name of the World Model instance.
        :type name: str
        :param actuation_config: Dictionary with the parameters of the actuation.
        :type actuation_config: dict
        :param perception_config: Dictionary with the parameters of the perception.
        :type perception_config: dict
        :param class_name: Name of the base WorldModel class, defaults to 'cognitive_nodes.world_model.WorldModel'.
        :type class_name: str
        """        
        super().__init__(name, class_name, **params)
        self.learner=Sim2D(actuation_config, perception_config, self.get_logger())
    
    def predict(self, perception, action):
        """
        Predicts the next perception according to a perception and an action.

        :param perception: The start perception.
        :type perception: cognitive_node_interfaces.msg.Perception
        :param action: The action performed.
        :type action: cognitive_node_interfaces.msg.Actuation
        :return: The predicted perception.
        :rtype: cognitive_node_interfaces.msg.Perception
        """        
        prediction=self.learner.predict(perception, action)
        return prediction
    

class SimBartender(WorldModel):
    """SimBartender class: A fixed world model of a bartender simulation."""
    def __init__(self, name='world_model', actuation_config=None, perception_config=None, class_name='cognitive_nodes.world_model.WorldModel', **params):
        """
        Constructor of the SimBartender class.

        :param name: The name of the World Model instance.
        :type name: str
        :param actuation_config: Dictionary with the parameters of the actuation.
        :type actuation_config: dict
        :param perception_config: Dictionary with the parameters of the perception.
        :type perception_config: dict
        :param class_name: Name of the base WorldModel class, defaults to 'cognitive_nodes.world_model.WorldModel'.
        :type class_name: str
        """
        super().__init__(name, class_name, **params)
    
    def calculate_activation(self, perception = None, activation_list=None):
        """
        Returns the activation value of the Model.

        :param perception: Perception does not influence the activation.
        :type perception: dict
        :param activation_list: List of activation values from other sources, defaults to None.
        :type activation_list: list
        :return: The activation of the instance and its timestamp.
        :rtype: cognitive_node_interfaces.msg.Activation
        """
        self.activation.timestamp = self.get_clock().now().to_msg()
        return self.activation
    
    def create_activation_input(self, node: dict): #Adds or deletes a node from the activation inputs list. By default reads activations.
        """
        Adds perceptions to the activation inputs list.

        :param node: Dictionary with the information of the node {'name': <name>, 'node_type': <node_type>}.
        :type node: dict
        """    
        name=node['name']
        node_type=node['node_type']
        if node_type == "Perception":
            subscriber=self.create_subscription(PerceptionStamped, "perception/" + str(name) + "/value", self.read_activation_callback, 1, callback_group=self.cbgroup_activation)
            data=Perception()
            updated=False
            timestamp=Time()
            new_input=dict(subscriber=subscriber, data=data, updated=updated, timestamp=timestamp)
            self.activation_inputs[name]=new_input
            self.get_logger().debug(f'{self.name} -- Created new activation input: {name} of type {node_type}')


    def read_activation_callback(self, msg: PerceptionStamped):
        """
        Callback method that reads a perception and stores it in the activation inputs list.

        :param msg: PerceptionStamped message that contains the perception and its timestamp.
        :type msg: cognitive_node_interfaces.msg.PerceptionStamped
        """        
        perception_dict=perception_msg_to_dict(msg=msg.perception)
        if len(perception_dict)>1:
            self.get_logger().error(f'{self.name} -- Received perception with multiple sensors: ({perception_dict.keys()}). Perception nodes should (currently) include only one sensor!')
        if len(perception_dict)==1:
            node_name=list(perception_dict.keys())[0]
            if node_name in self.activation_inputs:
                self.activation_inputs[node_name]['data']=perception_dict[node_name]
                self.activation_inputs[node_name]['updated']=True
                self.activation_inputs[node_name]['timestamp']=Time.from_msg(msg.timestamp)
        else:
            self.get_logger().warn("Empty perception recieved in P-Node. No activation calculated")

    
class Sim2D(Learner):
    """
    Sim2D class: A class that mimics a model that learned the dynamics of a 2D simulator.
    Actually it uses the same simulator as the environment to predict the next perception.
    """    
    def __init__(self, actuation_config, perception_config, logger:RcutilsLogger, **params):
        """
        Constructor of the Sim2D class.

        :param actuation_config: Dictionary with the parameters of the actuation.
        :type actuation_config: dict
        :param perception_config: Dictionary with the parameters of the perception.
        :type perception_config: dict
        :param logger: Logger object from the parent node.
        :type logger: RcutilsLogger
        """        
        super().__init__(None, **params)
        self.model=SimpleScenario(visualize=False)
        self.actuation_config=actuation_config
        self.perception_config=perception_config
        self.logger=logger

    def predict(self, perception: Perception, action: Actuation) -> Perception:  
        """
        Predicts the next perception according to a perception and an action.

        :param perception: The start perception.
        :type perception: cognitive_node_interfaces.msg.Perception
        :param action: The action performed.
        :type action: cognitive_node_interfaces.msg.Actuation
        :return: The predicted perception.
        :rtype: cognitive_node_interfaces.msg.Perception
        """        
        """"""
        perc_dict=self.denormalize(perception_msg_to_dict(perception), self.perception_config)
        act_dict=self.denormalize(actuation_msg_to_dict(action), self.actuation_config)
        
        self.logger.info(f"DEBUG: Perception {perc_dict}")
        self.logger.info(f"DEBUG: Action: {act_dict}")

        angle_l = act_dict["left_arm"][0]["angle"]
        angle_r = act_dict["right_arm"][0]["angle"]
        vel_l = act_dict["left_arm"][0]["dist"]
        vel_r = act_dict["right_arm"][0]["dist"]        
        gripper_l=perc_dict["ball_in_left_hand"][0]["data"]
        gripper_r=perc_dict["ball_in_right_hand"][0]["data"]
        #Set simulator to initial state:
        self.model.baxter_left.set_pos(perc_dict["left_arm"][0]["x"],perc_dict["left_arm"][0]["y"])
        self.model.baxter_left.set_angle(perc_dict["left_arm"][0]["angle"])
        self.model.baxter_left.set_gripper(gripper_l)
        self.model.baxter_right.set_pos(perc_dict["right_arm"][0]["x"],perc_dict["right_arm"][0]["y"])
        self.model.baxter_right.set_angle(perc_dict["right_arm"][0]["angle"])
        self.model.baxter_right.set_gripper(gripper_r)
        self.model.box1.set_pos(perc_dict["box"][0]["x"], perc_dict["box"][0]["y"])
        self.model.objects[0].set_pos(perc_dict["ball"][0]["x"], perc_dict["ball"][0]["y"])
        self.model.world_rules()

        #Apply action
        self.model.apply_action(angle_l, angle_r, vel_l, vel_r, gripper_l, gripper_r)

        #GRASP OBJECT IF GRIPPER IS CLOSE
        grippers_close = self.model.filter_entities(self.model.get_close_entities(self.model.robots[0], threshold=50), EntityType.ROBOT)
        self.logger.info(f"DEBUG - {[ent.name for ent in grippers_close]}")
        if grippers_close and not self.changed_grippers: #If grippers are close, change hands
            self.logger.info(f"DEBUG - Checking if changing grippers is possible")
            #Ball in left gripper
            if self.model.robots[0].catched_object and not self.model.robots[1].catched_object:
                gripper_l=False
                self.model.apply_action(gripper_left=gripper_l, gripper_right=gripper_r)
                gripper_r=True
                self.model.apply_action(gripper_left=gripper_l, gripper_right=gripper_r)
                self.changed_grippers=True

            #Ball in right gripper, optionale grippers
            self.logger.info(f"DEBUG - Checking if objects are close to gripper")
            self.changed_grippers=False
            close_l_obj = self.model.filter_entities(self.model.get_close_entities(self.model.robots[0], threshold=50), EntityType.BALL)
            close_r_obj = self.model.filter_entities(self.model.get_close_entities(self.model.robots[1], threshold=50), EntityType.BALL)
            if close_l_obj:
                self.logger.info(f"DEBUG - Objects {[obj.name for obj in close_l_obj]} detected close to left gripper")
                gripper_l = True
            if close_r_obj:
                self.logger.info(f"DEBUG - Objects {[obj.name for obj in close_r_obj]} detected close to right gripper")
                gripper_r = True
        
            #RELEASE OBJECT IF OVER BOX
            left_over_box = self.model.filter_entities(self.model.get_close_entities(self.model.robots[0], threshold=50), EntityType.BOX)
            right_over_box = self.model.filter_entities(self.model.get_close_entities(self.model.robots[1], threshold=50), EntityType.BOX)
            if left_over_box:
                self.logger.info(f"DEBUG - Boxes {[box.name for box in left_over_box]} detected close to left gripper")
                gripper_l = False
            if right_over_box:
                self.logger.info(f"DEBUG - Boxes {[box.name for box in right_over_box]} detected close to right gripper")
                gripper_r = False
            
            self.model.apply_action(gripper_left=gripper_l, gripper_right=gripper_r)

        #Read predicted perceptions
        left_arm=self.model.baxter_left.get_pos()
        left_angle=self.model.baxter_left.get_angle()
        right_arm=self.model.baxter_right.get_pos()
        right_angle=self.model.baxter_right.get_angle()
        ball=self.model.objects[0].get_pos()
        box=self.model.box1.get_pos()
        left_gripper= bool(self.model.baxter_left.catched_object)
        right_gripper= bool(self.model.baxter_right.catched_object)

        perc_dict["left_arm"][0]["x"] = float(left_arm[0])
        perc_dict["left_arm"][0]["y"] = float(left_arm[1])
        perc_dict["left_arm"][0]["angle"] = float(left_angle)
        perc_dict["ball_in_left_hand"][0]["data"] = left_gripper

        perc_dict["right_arm"][0]["x"] = float(right_arm[0])
        perc_dict["right_arm"][0]["y"] = float(right_arm[1])
        perc_dict["right_arm"][0]["angle"] = float(right_angle)
        perc_dict["ball_in_right_hand"][0]["data"] = right_gripper

        perc_dict["box"][0]["x"] = float(box[0])
        perc_dict["box"][0]["y"] = float(box[1])

        perc_dict["ball"][0]["x"] = float(ball[0])
        perc_dict["ball"][0]["y"] = float(ball[1])

        return perception_dict_to_msg(self.normalize(perc_dict, self.perception_config))

    def denormalize(self, input_dict, config):
        """
        Denormalize the input dictionary according to the configuration

        :param input_dict: Perception or actuation dictionary.
        :type input_dict: dict
        :param config: Configuration of the perception or actuation bounds.
        :type config: dict
        :return: Denormalized dictionary.
        :rtype: dict
        """        
        out=deepcopy(input_dict)
        for dim in out:
            for param in out[dim][0]:
                if config[dim][param]["type"]=="float":
                    bounds=config[dim][param]["bounds"]
                    value=out[dim][0][param]
                    out[dim][0][param]=bounds[0]+(value*(bounds[1]-bounds[0]))
        return out

    def normalize(self, input_dict, config):
        """
        Normalize the input dictionary according to the configuration.

        :param input_dict: Perception or actuation dictionary.
        :type input_dict: dict
        :param config: Configuration of the perception or actuation bounds.
        :type config: dict
        :return: Normalized dictionary.
        :rtype: dict
        """        
        out=deepcopy(input_dict)
        for dim in out:
            for param in out[dim][0]:
                if config[dim][param]["type"]=="float":
                    bounds=config[dim][param]["bounds"]
                    value=out[dim][0][param]
                    out[dim][0][param] = (value - bounds[0]) / (bounds[1] - bounds[0])
        return out
    

def main(args=None):
    rclpy.init(args=args)

    world_model = WorldModel()

    rclpy.spin(world_model)

    world_model.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()