import rclpy
import numpy as np
from copy import deepcopy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup


from cognitive_nodes.deliberative_model import DeliberativeModel, Learner, ANNLearner, Evaluator
from cognitive_nodes.episodic_buffer import EpisodicBuffer
from simulators.scenarios_2D import SimpleScenario, EntityType
from cognitive_node_interfaces.msg import Perception, Actuation, SuccessRate
from core.utils import actuation_dict_to_msg, actuation_msg_to_dict, perception_dict_to_msg, perception_msg_to_dict
from rclpy.impl.rcutils_logger import RcutilsLogger
from cognitive_nodes.episode import Episode, Action, episode_msg_to_obj, episode_msg_list_to_obj_list, episode_obj_list_to_msg_list 

from cognitive_node_interfaces.msg import Episode as EpisodeMsg


class WorldModel(DeliberativeModel):
    """
    World Model class: A static world model that is always active
    """
    def __init__(self, name='world_model', class_name = 'cognitive_nodes.world_model.WorldModel', episodes_topic=None, prediction_srv_type="cognitive_node_interfaces.srv.Predict", **params):
        """
        Constructor of the World Model class.

        Initializes a World Model instance with the given name and registers it in the LTM.

        :param name: The name of the World Model instance.
        :type name: str
        :param class_name: The name of the World Model class.
        :type class_name: str
        """
        super().__init__(name, class_name, node_type="world_model", prediction_srv_type="cognitive_node_interfaces.srv.Predict", **params)
        self.episodic_buffer=None
        self.learner=None
        self.confidence_evaluator=None
        self.activation.activation = 1.0

    def predict(self, input_episodes: list[Episode]) -> list[Episode]:
        self.get_logger().warning("The base WorldModel class does not implement any prediction. Returning the input episodes.")
        #self.get_logger().info(f"DEBUG input episodes - {input_episodes}")
        output_episodes = [Episode(perception=deepcopy(episode.old_perception), action=deepcopy(episode.action)) for episode in input_episodes]
        return output_episodes


class WorldModelLearned(WorldModel):
    """
    WorldModelLearned class: A world model that uses episodes to learn the dynamics of the world.
    """
    def __init__(self, name='world_model', class_name='cognitive_nodes.world_model.WorldModel', episodes_topic=None, retrain=True, **params):
        """
        Constructor of the WorldModelLearned class.

        :param name: The name of the World Model instance.
        :type name: str
        :param class_name: The name of the World Model class.
        :type class_name: str
        :param episodes_topic: The topic to subscribe to for episodes.
        :type episodes_topic: str
        """
        super().__init__(name, class_name, **params)
        self.cbgroup_episodes = MutuallyExclusiveCallbackGroup()
        
        self.episodes_topic = episodes_topic
        if self.episodes_topic is None:
            raise ValueError("episodes_topic must be provided for WorldModelLearned")

        self.episode_subscription = self.create_subscription(
            EpisodeMsg,
            self.episodes_topic,
            self.episode_callback,
            10,
            callback_group=self.cbgroup_episodes
        )

        self.episodic_buffer = EpisodicBuffer(
            node = self,
            main_size= 200,
            secondary_size= 50,
            train_split= 0.80,
            inputs = ["old_perception", "action"],
            outputs = ["perception"],
        )

        self.learner = ANNLearner(self, self.episodic_buffer, **params)

        self.confidence_evaluator = EvaluatorWorldModel(self, self.learner, self.episodic_buffer)

        self.retrain = retrain

    def predict(self, input_episodes: list[Episode]) -> list[Episode]:
        if not self.episodic_buffer.input_labels or not self.episodic_buffer.output_labels:
            self.get_logger().warning("Episodic buffer input or output labels are not defined. Returning the input episodes.")
            output_episodes = [Episode(perception=deepcopy(episode.old_perception), action=deepcopy(episode.action)) for episode in input_episodes]
            return output_episodes
        input_data = self.episodic_buffer.buffer_to_matrix(input_episodes, self.episodic_buffer.input_labels)
        self.get_logger().info(f"Data for prediction: {input_data.shape} samples. Ex: {input_data[:2]}")
        predictions = self.learner.call(input_data)
        if predictions is None:
            for episode in input_episodes:
                episode.perception = episode.old_perception  # If the model is not configured, return the old perception
            predicted_episodes = input_episodes  # If the model is not configured, return the input episodes
        else:
            self.get_logger().debug(f"Predictions: {predictions}")
            self.get_logger().debug(f"Output labels: {self.episodic_buffer.output_labels}")
            predicted_episodes = self.episodic_buffer.matrix_to_buffer(predictions, self.episodic_buffer.output_labels)
        self.get_logger().info(f"Prediction made: {len(predicted_episodes)} episodes")
        return predicted_episodes
    

    def episode_callback(self, msg: EpisodeMsg):
            """
            Callback for the episode subscription. It receives an episode message and adds it to the episodic buffer.

            :param msg: The episode message received.
            :type msg: cognitive_node_interfaces.msg.Episode
            """
            episode = episode_msg_to_obj(msg)
            self.episodic_buffer.add_episode(episode)
            self.get_logger().info(f"Episode added to buffer \n New train samples: {self.episodic_buffer.new_sample_count_main}, New test samples: {self.episodic_buffer.new_sample_count_secondary}")

            # TODO: Allow to train the learner in different moments, not only when the main buffer is full
            # If the main buffer is full, train the learner
            if self.episodic_buffer.new_sample_count_main >= self.episodic_buffer.main_max_size:
                if not self.learner.configured or self.retrain:   
                    self.get_logger().info("Training the learner with the new episodes")
                    x_train, y_train = self.episodic_buffer.get_train_samples(shuffle=True)
                    self.learner.train(x_train, y_train)
                    self.episodic_buffer.reset_new_sample_count(main=True, secondary=False)
                    self.get_logger().info("Learner trained with new episodes")

            if self.episodic_buffer.new_sample_count_secondary >= self.episodic_buffer.secondary_max_size and self.learner.configured:
                self.get_logger().info("Evaluating the learner with the new episodes")
                self.confidence_evaluator.evaluate()
                self.confidence_evaluator.publish_prediction_error()
                self.episodic_buffer.reset_new_sample_count(main=False, secondary=True)
                self.get_logger().info("Learner evaluated with new episodes")

    
    
class EvaluatorWorldModel(Evaluator):
    """
    EvaluatorWorldModel class: Evaluates the success rate of the world model based on its predictions.
    """
    def __init__(self, node:WorldModelLearned, learner:ANNLearner,  buffer:EpisodicBuffer, **params) -> None:
        """
        Constructor of the EvaluatorWorldModel class.

        :param learner: The learner to evaluate.
        :type learner: Learner
        :param buffer: Episodic buffer to use.
        :type buffer: EpisodicBuffer
        """        
        super().__init__(node, learner, buffer, **params)
        self.prediction_error = 0.0
        self.prediction_error_publisher = self.node.create_publisher(SuccessRate, f"world_model/{self.node.name}/prediction_error", 0)

    def evaluate(self):
        x_test, y_test = self.buffer.get_test_samples()
        self.prediction_error = self.learner.evaluate(x_test, y_test)
        self.node.get_logger().info(f"World Model Prediction Error: {self.prediction_error}")

    def publish_prediction_error(self):
        """
        Publishes the prediction error of the world model.
        """
        prediction_error_msg = SuccessRate()
        prediction_error_msg.node_name = self.node.name
        prediction_error_msg.node_type = self.node.node_type
        prediction_error_msg.success_rate = self.prediction_error
        self.prediction_error_publisher.publish(prediction_error_msg)



    


class Sim2DWorldModel(WorldModel):
    """
    Sim2DWorldModel class: A fixed world model of a 2D simulator. It uses the SimpleScenario simulator to predict the next perception.
    """    
    def __init__(self, name='world_model', wm_actuation_config=None, wm_perception_config=None, class_name='cognitive_nodes.world_model.WorldModel', **params):
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
        self.learner=Sim2D(self, wm_actuation_config, wm_perception_config, self.get_logger())

    def predict(self, input_episodes: list[Episode]) -> list[Episode]:
        predicted_episodes = [Episode(perception=self.learner.call(episode.old_perception, episode.action.actuation)) for episode in input_episodes]
        return predicted_episodes

    
class Sim2D(Learner):
    """
    Sim2D class: A class that mimics a model that learned the dynamics of a 2D simulator.
    Actually it uses the same simulator as the environment to predict the next perception.
    """    
    def __init__(self, node, actuation_config, perception_config, logger:RcutilsLogger, **params):
        """
        Constructor of the Sim2D class.

        :param actuation_config: Dictionary with the parameters of the actuation.
        :type actuation_config: dict
        :param perception_config: Dictionary with the parameters of the perception.
        :type perception_config: dict
        :param logger: Logger object from the parent node.
        :type logger: RcutilsLogger
        """        
        super().__init__(node, None, **params)
        self.model=SimpleScenario(visualize=False)
        self.changed_grippers = False
        self.actuation_config=actuation_config
        self.perception_config=perception_config
        self.logger=logger

    

    def call(self, perception, action) -> Perception:  
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
        self.logger.debug(f"DEBUG SIM2D: Perception: {perception} --- Action: {action}")
        perc_dict=self.denormalize(perception, self.perception_config)
        act_dict=self.denormalize(action, self.actuation_config)
        
        self.logger.debug(f"DEBUG: Perception {perc_dict}")
        self.logger.debug(f"DEBUG: Action: {act_dict}")

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
        grippers_close = self.model.filter_entities(self.model.get_close_entities(self.model.robots[0], threshold=250), EntityType.ROBOT)
        self.logger.debug(f"DEBUG - {[ent.name for ent in grippers_close]}")
        if grippers_close and not self.changed_grippers and (self.model.robots[0].catched_object or self.model.robots[1].catched_object): #If grippers are close, change hands
            self.logger.debug(f"DEBUG - Checking if changing grippers is possible")
            #Ball in left gripper
            if self.model.robots[0].catched_object and not self.model.robots[1].catched_object:
                gripper_l=False
                self.model.apply_action(gripper_left=gripper_l, gripper_right=gripper_r)
                self.model.objects[0].set_pos(*self.model.robots[1].get_pos()) #Move the ball to the right gripper
                gripper_r=True
                self.model.apply_action(gripper_left=gripper_l, gripper_right=gripper_r)
                self.changed_grippers=True
                self.logger.debug(f"DEBUG - Change from left to right gripper")

            #Ball in right gripper
            if self.model.robots[1].catched_object and not self.model.robots[0].catched_object:
                gripper_r=False
                self.model.apply_action(gripper_left=gripper_l, gripper_right=gripper_r)
                self.model.objects[0].set_pos(*self.model.robots[0].get_pos()) #Move the ball to the left gripper
                gripper_l=True
                self.model.apply_action(gripper_left=gripper_l, gripper_right=gripper_r)
                self.changed_grippers=True
                self.logger.debug(f"DEBUG - Change from right to left gripper")
            
        if not grippers_close: #Check if objects are close to the grippers
            self.logger.debug(f"DEBUG - Checking if objects are close to gripper")
            self.changed_grippers=False
            close_l_obj = self.model.filter_entities(self.model.get_close_entities(self.model.robots[0], threshold=50), EntityType.BALL)
            close_r_obj = self.model.filter_entities(self.model.get_close_entities(self.model.robots[1], threshold=50), EntityType.BALL)
            if close_l_obj:
                self.logger.debug(f"DEBUG - Objects {[obj.name for obj in close_l_obj]} detected close to left gripper")
                gripper_l = True
            if close_r_obj:
                self.logger.debug(f"DEBUG - Objects {[obj.name for obj in close_r_obj]} detected close to right gripper")
                gripper_r = True
        
            #RELEASE OBJECT IF OVER BOX
            left_over_box = self.model.filter_entities(self.model.get_close_entities(self.model.robots[0], threshold=50), EntityType.BOX)
            right_over_box = self.model.filter_entities(self.model.get_close_entities(self.model.robots[1], threshold=50), EntityType.BOX)
            if left_over_box:
                self.logger.debug(f"DEBUG - Boxes {[box.name for box in left_over_box]} detected close to left gripper")
                gripper_l = False
            if right_over_box:
                self.logger.debug(f"DEBUG - Boxes {[box.name for box in right_over_box]} detected close to right gripper")
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

        return self.normalize(perc_dict, self.perception_config)

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
        for dim in input_dict:
            for param in input_dict[dim][0]:
                config_item = config[dim].get(param, {"type": None})
                if config_item["type"]=="float":
                    bounds=config[dim][param]["bounds"]
                    value=out[dim][0][param]
                    out[dim][0][param]=bounds[0]+(value*(bounds[1]-bounds[0]))
                if config_item["type"] is None:
                    if param == "angle_cos":
                        continue
                    if param == "angle_sin":
                        bounds=config[dim]["angle"]["bounds"]
                        angle_sin = out[dim][0]["angle_sin"]*2 - 1.0  # Denormalize from [0, 1] to [-1, 1]
                        angle_cos = out[dim][0]["angle_cos"]*2 - 1.0  # Denormalize from [0, 1] to [-1, 1]
                        angle_rad = np.arctan2(angle_sin, angle_cos)
                        if bounds == [-180, 180]:
                            angle_deg = angle_rad * (180.0 / np.pi)
                            out[dim][0]["angle"] = angle_deg
                        else:
                            out[dim][0]["angle"] = angle_rad
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
        for dim in input_dict:
            for param in input_dict[dim][0]:
                config_item = config[dim].get(param, {"type": None})
                if config_item["type"]=="float":
                    bounds=config_item["bounds"]
                    value=out[dim][0][param]
                    out[dim][0][param] = (value - bounds[0]) / (bounds[1] - bounds[0])
                if config_item["type"] == "angle":
                    # Check if angle is in degrees (common ranges: 0-360, -180 to 180)
                    angle_range = config[dim]["angle"]["bounds"][1] - config[dim]["angle"]["bounds"][0]
                    if angle_range > 2 * np.pi:  # Likely in degrees
                        angle_rad = out[dim][0]["angle"] * np.pi / 180.0
                    else:  # Already in radians
                        angle_rad = out[dim][0]["angle"]

                    angle_cos_raw = np.cos(angle_rad)
                    angle_sin_raw = np.sin(angle_rad)
                    # normalize from [-1, 1] to [0, 1] and clip to avoid tiny numerical drift
                    angle_cos = min(max((angle_cos_raw + 1.0) / 2.0, 0.0), 1.0)
                    angle_sin = min(max((angle_sin_raw + 1.0) / 2.0, 0.0), 1.0)
                    out[dim][0].pop("angle")  # Remove the original angle
                    out[dim][0]["angle_cos"] = angle_cos
                    out[dim][0]["angle_sin"] = angle_sin
        return out
    

def main(args=None):
    rclpy.init(args=args)

    world_model = WorldModel()

    rclpy.spin(world_model)

    world_model.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()