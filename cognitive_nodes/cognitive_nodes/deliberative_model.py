import rclpy
from rclpy.impl.rcutils_logger import RcutilsLogger

import tensorflow as tf
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.losses import Loss
from keras._tf_keras.keras.callbacks import TensorBoard
from keras import layers, metrics, losses, Sequential

from core.cognitive_node import CognitiveNode
from core.utils import class_from_classname, msg_to_dict
from cognitive_node_interfaces.srv import SetActivation, Predict, GetSuccessRate, IsCompatible
from cognitive_nodes.episodic_buffer import EpisodicBuffer
from cognitive_nodes.episode import Episode, Action, episode_msg_to_obj, episode_msg_list_to_obj_list, episode_obj_list_to_msg_list

class DeliberativeModel(CognitiveNode):
    """
    Deliberative Model class, this class is a generic model that can be used to implement different types of deliberative models. 
    """
    def __init__(self, name='model', class_name = 'cognitive_nodes.deliberative_model.DeliberativeModel', node_type="deliberative_model", prediction_srv_type=None, **params):
        """
        Constructor of the Deliberative Model class.

        Initializes a Deliberative instance with the given name.

        :param name: The name of the Deliberative Model instance.
        :type name: str
        :param class_name: The name of the DeliberativeModel class.
        :type class_name: str
        :param node_type: The type of the node, defaults to "deliberative_model".
        :type node_type: str
        """
        super().__init__(name, class_name, **params)

        self.episodic_buffer=None
        self.learner=None
        self.confidence_evaluator=None

        # N: Set Activation Service
        self.set_activation_service = self.create_service(
            SetActivation,
            node_type+ "/" + str(name) + '/set_activation',
            self.set_activation_callback,
            callback_group=self.cbgroup_server
        )

        # N: Predict Service
        if prediction_srv_type is not None:
            prediction_srv_type = class_from_classname(prediction_srv_type)
        else:
            raise ValueError("prediction_srv_type must be provided and be a valid class name.")
        self.predict_service = self.create_service(
            prediction_srv_type,
            node_type+ "/" + str(name) + '/predict',
            self.predict_callback,
            callback_group=self.cbgroup_server
        )

        # N: Get Success Rate Service
        self.get_success_rate_service = self.create_service(
            GetSuccessRate,
            node_type+ "/" + str(name) + '/get_success_rate',
            self.get_success_rate_callback,
            callback_group=self.cbgroup_server
        )

        # N: Is Compatible Service
        self.is_compatible_service = self.create_service(
            IsCompatible,
            node_type+ "/" + str(name) + '/is_compatible',
            self.is_compatible_callback,
            callback_group=self.cbgroup_server
        )

        #TODO: Set activation from main_loop
        #self.activation.activation = 1.0

    def set_activation_callback(self, request, response):
        """
        Some processes can modify the activation of a Model.

        :param request: The request that contains the new activation value.
        :type request: cognitive_node_interfaces.srv.SetActivation.Request
        :param response: The response indicating if the activation was set.
        :type response: cognitive_node_interfaces.srv.SetActivation.Response
        :return: The response indicating if the activation was set.
        :rtype: cognitive_node_interfaces.srv.SetActivation.Response
        """
        activation = request.activation
        self.get_logger().info('Setting activation ' + str(activation) + '...')
        self.activation.activation = activation
        self.activation.timestamp = self.get_clock().now().to_msg()
        response.set = True
        return response
    
    def predict_callback(self, request, response):
        """
        Get predicted perception values for the last perceptions not newer than a given
        timestamp and for a given policy.

        :param request: The request that contains the timestamp and the policy.
        :type request: cognitive_node_interfaces.srv.Predict.Request
        :param response: The response that included the obtained perception.
        :type response: cognitive_node_interfaces.srv.Predict.Response
        :return: The response that included the obtained perception.
        :rtype: cognitive_node_interfaces.srv.Predict.Response
        """
        self.get_logger().info('Predicting ...') 
        input_episodes = episode_msg_list_to_obj_list(request.input_episodes)
        output_episodes = self.predict(input_episodes)
        response.output_episodes = episode_obj_list_to_msg_list(output_episodes)
        self.get_logger().info(f"Prediction made... ")
        return response
    
    def get_success_rate_callback(self, request, response): # TODO: implement
        """
        Get a prediction success rate based on a historic of previous predictions.

        :param request: Empty request.
        :type request: cognitive_node_interfaces.srv.GetSuccessRate.Request
        :param response: The response that contains the predicted success rate.
        :type response: cognitive_node_interfaces.srv.GetSuccessRate.Response
        :return: The response that contains the predicted success rate.
        :rtype: cognitive_node_interfaces.srv.GetSuccessRate.Response
        """
        self.get_logger().info('Getting success rate..')
        raise NotImplementedError
        response.success_rate = 0.5
        return response
    
    def is_compatible_callback(self, request, response): # TODO: implement
        """
        Check if the Model is compatible with the current available perceptions.

        :param request: The request that contains the current available perceptions.
        :type request: cognitive_node_interfaces.srv.IsCompatible.Request
        :param response: The response indicating if the Model is compatible or not.
        :type response: cognitive_node_interfaces.srv.IsCompatible.Response
        :return: The response indicating if the Model is compatible or not.
        :rtype: cognitive_node_interfaces.srv.IsCompatible.Response
        """
        self.get_logger().info('Checking if compatible..')
        raise NotImplementedError
        response.compatible = True
        return response

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

    def predict(self, input_episodes: list[Episode]) -> list:
        input_data = self.episodic_buffer.buffer_to_matrix(input_episodes, self.episodic_buffer.input_labels)
        predictions = self.learner.call(input_data)
        if predictions is None:
            predicted_episodes = input_episodes  # If the model is not configured, return the input episodes
        else:
            self.get_logger().info(f"Predictions: {predictions}")
            self.get_logger().info(f"Output labels: {self.episodic_buffer.output_labels}")
            predicted_episodes = self.episodic_buffer.matrix_to_buffer(predictions, self.episodic_buffer.output_labels)
        self.get_logger().info(f"Prediction made: {predicted_episodes}")
        return predicted_episodes
    
    
    
class Learner:
    """
    Class that wraps around a learning model (Linear Classifier, ANN, SVM...)
    """    
    def __init__(self, node:CognitiveNode, buffer:EpisodicBuffer, **params) -> None:
        """
        Constructor of the Learner class.

        :param buffer: Episodic buffer to use.
        :type buffer: generic_model.EpisodicBuffer
        """        
        self.node = node
        self.model=None
        self.buffer=buffer
        self.configured=False

    def train(self):
        """
        Placeholder method for training the model.

        :raises NotImplementedError: Not implemented yet.
        """        
        raise NotImplementedError
    
    def call(self, x):
        """
        Placeholder method for predicting an outcome.

        :param perception: Perception dictionary.
        :type perception: dict
        :param action: Candidate action dictionary.
        :type action: dict
        :raises NotImplementedError: Not implemented yet.
        """        
        raise NotImplementedError
    

class ANNLearner(Learner):
    def __init__(self, node, buffer, batch_size=32, epochs=50, output_activation='sigmoid', hidden_activation='relu', hidden_layers=[128], learning_rate=0.001, model_file=None, tensorboard=False, tensorboard_log_dir=None, **params):
        super().__init__(node, buffer, **params)
        tf.config.set_visible_devices([], 'GPU') # TODO: Handle GPU usage properly
        self.batch_size = batch_size
        self.epochs = epochs
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.configured = False
        self.model_file = model_file
        self.tensorboard = tensorboard
        self.tensorboard_log_dir = tensorboard_log_dir
        self._run_counter = 0
        if self.model_file is not None:
            self.configure_model(0,0) # Load model from file

    def configure_model(self, input_length, output_length):
        """
        Configure the ANN model with the given input shape, output shape, and labels.

        :param input_shape: The shape of the input data.
        :type input_shape: int
        :param output_shape: The shape of the output data.
        :type output_shape: int
        """
        if self.model_file is None:
            self.model = Sequential()
            
            ## TODO: USE THE LABELS TO SEPARATE THE INPUTS INTO REGUAR INPUTS AND THE POLICY ID INPUT, THEN CONCATNATE ##
            # TODO: THIS MIGHT REQUIRE TO USE THE FUNCTIONAL API INSTEAD OF SEQUENTIAL
            # --- Inputs ---
            # object_input = layers.Input(shape=(), dtype=tf.int32, name="object_id")
            # numeric_input = layers.Input(shape=(num_numeric_features,), dtype=tf.float32, name="numeric_features")

            # --- Embedding Layer ---
            #embedding_layer = layers.Embedding(input_dim=num_objects, output_dim=embedding_dim)
            #embedded_object = embedding_layer(object_input)  # shape: (batch_size, embedding_dim)

            ## TODO: USE THE LABELS TO SEPARATE THE INPUTS INTO REGUAR INPUTS AND THE POLICY ID INPUT, THEN CONCATNATE ##

            self.model.add(layers.Input(shape=(input_length,)))
            for units in self.hidden_layers:
                self.model.add(layers.Dense(units,
                                        activation=self.hidden_activation,
                                        kernel_initializer="he_normal",
                                        kernel_regularizer=tf.keras.regularizers.l2(1e-5)))
                self.model.add(layers.BatchNormalization())   # stabilizes training
                self.model.add(layers.Dropout(0.1))           # mild dropout
            self.model.add(layers.Dense(output_length, activation=self.output_activation))
            #self.model.compile(optimizer=self.optimizer, loss=AsymmetricMSE(underestimation_penalty=3.0), metrics=['mae'])
            self.model.compile(optimizer=self.optimizer, loss="mse", metrics=['mae'])
            self.configured = True
        else:
            self.node.get_logger().info(f"Loading model from {self.model_file}")
            self.model = tf.keras.models.load_model(self.model_file, custom_objects={"AsymmetricMSE": AsymmetricMSE})
            self.configured = True               
    
    def reset_optimizer_state(self):
        """Reset only the optimizer state, keeping model weights unchanged."""
        if self.configured:
            # Create a fresh optimizer instance
            fresh_optimizer = Adam(learning_rate=self.learning_rate)
            # Recompile with fresh optimizer (weights unchanged)
            self.model.compile(optimizer=fresh_optimizer, loss="mse", metrics=['mae'])

    def train(self, x_train, y_train, epochs=None, batch_size=None, validation_split=0.0, x_val=None, y_val=None, verbose=1, reset_optimizer=True):
        # Ensure x_train and y_train are at least 2D
        if len(x_train.shape) == 1:
            x_train = x_train.reshape(-1, 1)
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)

        # Handle validation data if provided
        validation_data = None
        if x_val is not None and y_val is not None:
            # Ensure x_val and y_val are at least 2D
            if len(x_val.shape) == 1:
                x_val = x_val.reshape(-1, 1)
            if len(y_val.shape) == 1:
                y_val = y_val.reshape(-1, 1)
            validation_data = (x_val, y_val)
            # If validation_data is provided, set validation_split to 0.0
            validation_split = 0.0

        
        if not epochs:
            epochs = self.epochs
        if not batch_size:
            batch_size = self.batch_size
        if not self.configured:
            self.configure_model(x_train.shape[1], y_train.shape[1])
        callbacks = []

        if reset_optimizer:
            self.reset_optimizer_state()

        # Learning rate scheduling
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=epochs // 10,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(lr_scheduler)
        
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=epochs // 5,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)

        if self.tensorboard:
            if self.tensorboard_log_dir is None:
                self.tensorboard_log_dir = f'logs/fit/{self.node.name}'
            run_dict = self.tensorboard_log_dir + f'/run_{self._run_counter}'
            self._run_counter += 1
            tensorboard_callback = TensorBoard(log_dir=run_dict, histogram_freq=1)
            callbacks.append(tensorboard_callback)
        self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_split=validation_split,
            validation_data=validation_data,
            callbacks=callbacks
        )

    def call(self, x):
        if not self.configured:
            return None
        return self.model.predict(x)
    
    def evaluate(self, x_test, y_test):
        if not self.configured:
            return None
        return self.model.evaluate(x_test, y_test, verbose=0)[1]

class AsymmetricMSE(Loss):
    def __init__(self, underestimation_penalty=1.0, overestimation_penalty=1.0, name="asymmetric_mse"):
        """
        underestimation_penalty: float, multiplier applied when y_pred < y_true
        overestimation_penalty: float, multiplier applied when y_pred > y_true
        """
        super().__init__(name=name)
        self.underestimation_penalty = underestimation_penalty
        self.overestimation_penalty = overestimation_penalty

    def call(self, y_true, y_pred):
        error = y_pred - y_true
        weight = tf.where(error < 0, self.underestimation_penalty, self.overestimation_penalty)
        return tf.reduce_mean(weight * tf.square(error))

    def get_config(self):
        config = super().get_config()
        config.update({
            "underestimation_penalty": self.underestimation_penalty
        })
        return config

class Evaluator:
    """
    Class that evaluates the success rate of a model based on its predictions.
    """    
    def __init__(self, node:CognitiveNode, learner:Learner,  buffer:EpisodicBuffer, **params) -> None:
        """
        Constructor of the Learner class.

        :param node: Cognitive node that uses this model.
        :type node: CognitiveNode
        :param learner: Learner instance to use for predictions.
        :type learner: Learner
        :param buffer: Episodic buffer to use.
        :type buffer: generic_model.EpisodicBuffer
        """
        self.node = node
        self.learner = learner
        self.buffer = buffer
        self.learner = learner

    def evaluate(self):
        """
        Placeholder method for evaluating the model's success rate.

        :raises NotImplementedError: Not implemented yet.
        """
        raise NotImplementedError


