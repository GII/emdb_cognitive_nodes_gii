import rclpy
from copy import deepcopy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

import tensorflow as tf
from keras._tf_keras.keras.optimizers import Adam
from keras import layers, metrics, losses, Sequential

from cognitive_nodes.deliberative_model import DeliberativeModel, Learner, Evaluator
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
    def __init__(self, name='world_model', class_name = 'cognitive_nodes.world_model.WorldModel', episodes_topic=None, **params):
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

    def predict(self, perception: Perception, action: Actuation) -> Perception:
        return perception, True  # Default implementation, should be overridden by subclasses


class WorldModelLearned(WorldModel):
    """
    WorldModelLearned class: A world model that uses episodes to learn the dynamics of the world.
    """
    def __init__(self, name='world_model', class_name='cognitive_nodes.world_model.WorldModel', episodes_topic=None, **params):
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

        self.learner = ANNWorldModel(self.episodic_buffer)

        self.confidence_evaluator = EvaluatorWorldModel(self, self.learner, self.episodic_buffer)

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
            if self.episodic_buffer.new_sample_count_main >= self.episodic_buffer.main_size:
                self.get_logger().info("Training the learner with the new episodes")
                x_train, y_train = self.episodic_buffer.get_train_samples()
                self.learner.train(x_train, y_train)
                self.episodic_buffer.reset_new_sample_count(main=True, secondary=False)
                self.get_logger().info("Learner trained with new episodes")

            if self.episodic_buffer.new_sample_count_secondary >= self.episodic_buffer.secondary_size and self.learner.configured:
                self.get_logger().info("Evaluating the learner with the new episodes")
                self.confidence_evaluator.evaluate()
                self.confidence_evaluator.publish_prediction_error()
                self.episodic_buffer.reset_new_sample_count(main=False, secondary=True)
                self.get_logger().info("Learner evaluated with new episodes")

    def predict_callback(self, request, response): 
        """
        Gets the prediction of the world model based on the input episodes.

        :param request: The request that contains the timestamp and the policy.
        :type request: cognitive_node_interfaces.srv.Predict.Request
        :param response: The response that included the obtained perception.
        :type response: cognitive_node_interfaces.srv.Predict.Response
        :return: The response that included the obtained perception.
        :rtype: cognitive_node_interfaces.srv.Predict.Response
        """
        self.get_logger().info('Predicting ...')
        buffer = episode_msg_list_to_obj_list(request.input_episodes)
        self.get_logger().info(f"Input buffer: {buffer}")
        input_data = self.episodic_buffer.buffer_to_matrix(buffer, self.episodic_buffer.input_labels)
        predictions = self.learner.predict(input_data)
        if predictions is None:
            predicted_episodes = buffer  # If the model is not configured, return the input episodes
        else:
            self.get_logger().info(f"Predictions: {predictions}")
            self.get_logger().info(f"Output labels: {self.episodic_buffer.output_labels}")
            predicted_episodes = self.episodic_buffer.matrix_to_buffer(predictions, self.episodic_buffer.output_labels)
        response.output_episodes = episode_obj_list_to_msg_list(predicted_episodes)
        self.get_logger().info(f"Prediction made: {predicted_episodes}")
        return response

class ANNWorldModel(Learner):
    def __init__(self, buffer, **params):
        super().__init__(buffer, **params)

        self.batch_size = 50
        self.epochs = 50
        self.output_activation = 'relu'
        self.hidden_activation = 'relu'
        self.hidden_layers = [128]
        self.learning_rate = 0.001
        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.configured = False

    def configure_model(self, input_length, output_length):
        """
        Configure the ANN model with the given input shape, output shape, and labels.

        :param input_shape: The shape of the input data.
        :type input_shape: int
        :param output_shape: The shape of the output data.
        :type output_shape: int
        """
        self.model = Sequential()
        
        ## TODO: USE THE LABELS TO SEPARATE THE INPUTS INTO REGUAR INPUTS AND THE POLICY ID INPUT, THEN CONCATNATE ##
          #TODO: THIS MIGHT REQUIRE TO USE THE FUNCTIONAL API INSTEAD OF SEQUENTIAL
        # --- Inputs ---
        # object_input = layers.Input(shape=(), dtype=tf.int32, name="object_id")
        # numeric_input = layers.Input(shape=(num_numeric_features,), dtype=tf.float32, name="numeric_features")

        # --- Embedding Layer ---
        #embedding_layer = layers.Embedding(input_dim=num_objects, output_dim=embedding_dim)
        #embedded_object = embedding_layer(object_input)  # shape: (batch_size, embedding_dim)

        ## TODO: USE THE LABELS TO SEPARATE THE INPUTS INTO REGUAR INPUTS AND THE POLICY ID INPUT, THEN CONCATNATE ##

        self.model.add(layers.Input(shape=(input_length,)))
        for units in self.hidden_layers:
            self.model.add(layers.Dense(units, activation=self.hidden_activation))
        self.model.add(layers.Dropout(0.2))  # Add dropout for regularization
        self.model.add(layers.Dense(output_length, activation=self.output_activation))
        self.model.compile(optimizer=self.optimizer, loss='mse', metrics=['mae'])
        self.configured = True               
    
    def train(self, x_train, y_train, epochs=100, batch_size=32, verbose=1):
        if not self.configured:
            self.configure_model(x_train.shape[1], y_train.shape[1])
        self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
            )
    
    def predict(self, x):
        if not self.configured:
            return None
        return self.model.predict(x)
    
    def evaluate(self, x_test, y_test):
        if not self.configured:
            return None
        return self.model.evaluate(x_test, y_test, verbose=0)[1]
    
class EvaluatorWorldModel(Evaluator):
    """
    EvaluatorWorldModel class: Evaluates the success rate of the world model based on its predictions.
    """
    def __init__(self, node:WorldModelLearned, learner:ANNWorldModel,  buffer:EpisodicBuffer, **params) -> None:
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