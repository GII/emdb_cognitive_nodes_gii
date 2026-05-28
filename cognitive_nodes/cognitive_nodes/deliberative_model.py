import os
import numpy as np


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


from core.cognitive_node import CognitiveNode
from core.container import Container
from core.utils import class_from_classname
from cognitive_node_interfaces.srv import SetActivation, GetSuccessRate, IsCompatible, SaveModel
from cognitive_nodes.episodic_buffer import EpisodicBuffer
from cognitive_nodes.episode import Episode, container_msg_to_episode, container_to_episode_obj

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
        :param params: Additional keyword parameters reserved for future use.
        :type params: dict
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

        # N: Save Model Service
        self.save_model_service = self.create_service(
            SaveModel,
            node_type+ "/" + str(name) + '/save_model',
            self.save_model_callback,
            callback_group=self.cbgroup_server
        )

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
        input_episodes = container_msg_to_episode(request.input_episodes)
        output_episodes = self.predict(input_episodes)
        response.output_episodes = container_msg_to_episode(output_episodes)
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

    def save_model_callback(self, request, response):
        """
        Save the current model to a file.

        :param request: The request that contains the prefix and suffix for the file name.
        :type request: cognitive_node_interfaces.srv.SaveModel.Request
        :param response: The response that contains the saved model path and success status.
        :type response: cognitive_node_interfaces.srv.SaveModel.Response
        :return: The response that contains the saved model path and success status.
        :rtype: cognitive_node_interfaces.srv.SaveModel.Response
        """
        self.get_logger().info('Saving model...')
        if self.learner is not None and hasattr(self.learner, 'save_model'):
            model_name = f"{request.prefix}{self.name}{request.suffix}"
            try:
                success, path = self.learner.save_model(model_name)
            except Exception as e:
                self.get_logger().error(f"Error saving model: {e}")
                path = ""
                success = False
            response.saved_model_path = path
            response.success = success
            if success:
                self.get_logger().info(f"Model saved to {path}.")
            else:
                self.get_logger().error("Failed to save model.")
        else:
            response.saved_model_path = ""
            response.success = False
            self.get_logger().error("Learner does not support saving models.")
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

    def predict(self, input_episodes: Container) -> Episode:
        input_data = self.episodic_buffer.buffer_to_matrix(input_episodes, self.episodic_buffer.input_labels)
        predictions = self.learner.call(input_data)
        if predictions is None:
            predicted_episodes = input_episodes  # If the model is not configured, return the input episodes
        else:
            self.get_logger().info(f"Predictions: {predictions}")
            self.get_logger().info(f"Output labels: {self.episodic_buffer.output_labels}")
            prediction_container = self.episodic_buffer.matrix_to_container(predictions, self.episodic_buffer.output_labels, name= "predicted_episodes")
            predicted_episodes = container_to_episode_obj(prediction_container)
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
        :type buffer: cognitive_nodes.episodic_buffer.EpisodicBuffer
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
    """
    Class that implements a Neural Network-based learner using PyTorch.
    """    


    def __init__(self, node, buffer, batch_size=32, epochs=50, output_activation='sigmoid', 
                 hidden_activation='relu', hidden_layers=[128], learning_rate=0.001, loss_function=nn.MSELoss, val_function=nn.L1Loss, 
                 model_file=None, device='cpu', **params):
        """Initialize the ANNLearner with PyTorch-based neural network configuration.

        :param node: The cognitive node that uses this learner.
        :type node: CognitiveNode
        :param buffer: The episodic buffer for storing and retrieving episodes.
        :type buffer: EpisodicBuffer
        :param batch_size: Number of samples per batch during training, defaults to 32
        :type batch_size: int, optional
        :param epochs: Maximum number of training epochs, defaults to 50
        :type epochs: int, optional
        :param output_activation: Activation function for output layer ('sigmoid', 'tanh', 'relu', 'linear'), defaults to 'sigmoid'
        :type output_activation: str, optional
        :param hidden_activation: Activation function for hidden layers ('relu', 'tanh', 'sigmoid'), defaults to 'relu'
        :type hidden_activation: str, optional
        :param hidden_layers: List of hidden layer sizes, defaults to [128]
        :type hidden_layers: list, optional
        :param learning_rate: Learning rate for Adam optimizer, defaults to 0.001
        :type learning_rate: float, optional
        :param loss_function: Loss function class for training, defaults to nn.MSELoss
        :type loss_function: type, optional
        :param val_function: Loss function class for validation, defaults to nn.L1Loss
        :type val_function: type, optional
        :param model_file: Path to pre-trained model file to load, defaults to None
        :type model_file: str, optional
        :param device: Device to use for training ('cpu' or 'cuda'), defaults to 'cpu'
        :type device: str, optional
        :param params: Additional keyword parameters reserved for future use.
        :type params: dict
        """
        super().__init__(node, buffer, **params)
        
        # PyTorch specific setup
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')

        self.batch_size = batch_size
        self.epochs = epochs
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.configured = False
        self.model_file = model_file
        self._run_counter = 0
        
        # Model and optimizer will be initialized later
        self.model = None
        self.optimizer = None
        self.criterion = loss_function()
        self.val_criterion = val_function()
        
        if self.model_file is not None:
            self.load_model()

    def configure_model(self, input_length, output_length):
        """Configure the ANN model architecture and initialize the optimizer.

        :param input_length: Number of input features for the neural network.
        :type input_length: int
        :param output_length: Number of output features/predictions from the neural network.
        :type output_length: int
        """
        self.model = ANNModel(
            input_size=input_length,
            output_size=output_length,
            hidden_layers=self.hidden_layers,
            hidden_activation=self.hidden_activation,
            output_activation=self.output_activation
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.input_length = input_length
        self.output_length = output_length
        self.configured = True
        
        self.node.get_logger().info(f"Model configured with input: {input_length}, output: {output_length}")

    def load_model(self):
        """Loads model from file."""
        if os.path.exists(self.model_file):
            checkpoint = torch.load(self.model_file, map_location=self.device)
            
            # Extract model configuration from checkpoint
            self.input_length = checkpoint['input_length']
            self.output_length = checkpoint['output_length']
            self.hidden_layers = checkpoint['hidden_layers']
            self.hidden_activation = checkpoint['hidden_activation']
            self.output_activation = checkpoint['output_activation']
            self.learning_rate = checkpoint['learning_rate']
            
            # Configure model architecture
            self.configure_model(self.input_length, self.output_length)
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            self.node.get_logger().info(f"Model loaded from {self.model_file}")
        else:
            self.node.get_logger().warning(f"Model file {self.model_file} not found")

    def save_model(self, filepath):
        """Save the trained model to a PyTorch checkpoint file.

        :param filepath: Path where the model checkpoint will be saved. Automatically adds '.pth' extension if not present.
        :type filepath: str
        :return: Tuple containing success status and the path where the model was saved.
        :rtype: tuple(bool, str)
        """        
        if not self.configured:
            self.node.get_logger().warning("Model not configured. Cannot save.")
            return False, ""
            
        filepath = filepath if filepath.endswith('.pth') else filepath + '.pth'
        if filepath is None:
            self.node.get_logger().warning("No save path provided")
            return False, ""
            
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_length': self.input_length,
            'output_length': self.output_length,
            'hidden_layers': self.hidden_layers,
            'hidden_activation': self.hidden_activation,
            'output_activation': self.output_activation,
            'learning_rate': self.learning_rate,
            'input_labels': self.buffer.input_labels,
            'output_labels': self.buffer.output_labels
        }
        
        torch.save(checkpoint, filepath)
        self.node.get_logger().info(f"Model saved to {filepath}")
        return True, filepath

    def reset_model_state(self):
        """Reset optimizer state while keeping model weights."""
        if self.configured:
            # Save current weights
            weights = self.model.state_dict().copy()
            
            # Reinitialize optimizer
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            # Restore weights
            self.model.load_state_dict(weights)

    def train(self, x_train, y_train, epochs=None, batch_size=None, validation_split=0.0, 
              x_val=None, y_val=None, verbose=1, reset_optimizer=True):
        """Train the neural network model using PyTorch with optional validation and early stopping.

        :param x_train: Training input data (features).
        :type x_train: np.ndarray or torch.Tensor
        :param y_train: Training target data (labels).
        :type y_train: np.ndarray or torch.Tensor
        :param epochs: Number of training epochs. If None, uses the learner's default epochs, defaults to None
        :type epochs: int, optional
        :param batch_size: Batch size for training. If None, uses the learner's default batch size, defaults to None
        :type batch_size: int, optional
        :param validation_split: Fraction of training data to use for validation (0.0 to 1.0), defaults to 0.0
        :type validation_split: float, optional
        :param x_val: Validation input data. If provided with y_val, overrides validation_split, defaults to None
        :type x_val: np.ndarray or torch.Tensor, optional
        :param y_val: Validation target data. If provided with x_val, overrides validation_split, defaults to None
        :type y_val: np.ndarray or torch.Tensor, optional
        :param verbose: Verbosity level. 0 = silent, 1 = progress logs, defaults to 1
        :type verbose: int, optional
        :param reset_optimizer: Whether to reset the optimizer state before training, defaults to True
        :type reset_optimizer: bool, optional
        """
        
        # Convert numpy arrays to torch tensors
        if isinstance(x_train, np.ndarray):
            x_train = torch.FloatTensor(x_train)
        if isinstance(y_train, np.ndarray):
            y_train = torch.FloatTensor(y_train)
            
        # Ensure proper dimensions
        if len(x_train.shape) == 1:
            x_train = x_train.unsqueeze(1)
        if len(y_train.shape) == 1:
            y_train = y_train.unsqueeze(1)
            
        epochs = epochs or self.epochs
        batch_size = batch_size or self.batch_size
        
        # Configure model if not done
        if not self.configured:
            self.configure_model(x_train.shape[1], y_train.shape[1])
            
        if reset_optimizer:
            self.reset_model_state()
            
        # Prepare validation data
        val_loader = None
        if x_val is not None and y_val is not None:
            if isinstance(x_val, np.ndarray):
                x_val = torch.FloatTensor(x_val)
            if isinstance(y_val, np.ndarray):
                y_val = torch.FloatTensor(y_val)
                
            if len(x_val.shape) == 1:
                x_val = x_val.unsqueeze(1)
            if len(y_val.shape) == 1:
                y_val = y_val.unsqueeze(1)
                
            val_dataset = TensorDataset(x_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        elif validation_split > 0:
            # Split training data for validation
            split_idx = int(len(x_train) * (1 - validation_split))
            x_val = x_train[split_idx:]
            y_val = y_train[split_idx:]
            x_train = x_train[:split_idx]
            y_train = y_train[:split_idx]
            
            val_dataset = TensorDataset(x_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
        # Create data loader for training
        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, 
            patience=epochs//10, min_lr=1e-7,
        )
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = epochs // 5
        best_epoch = 0
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            train_loss = 0.0
            num_batches = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
                
            avg_train_loss = train_loss / num_batches
            
            # Validation
            val_loss = None
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                val_batches = 0
                
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        outputs = self.model(batch_x)
                        loss = self.val_criterion(outputs, batch_y)
                        val_loss += loss.item()
                        val_batches += 1
                        
                val_loss = val_loss / val_batches
                self.model.train()
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best weights
                    self.best_weights = self.model.state_dict().copy()
                    best_epoch = epoch
                else:
                    patience_counter += 1
                    
                if patience_counter >= early_stopping_patience:
                    if verbose > 0:
                        self.node.get_logger().info(f"Early stopping at epoch {epoch+1}. Restoring weights from epoch {best_epoch+1} with val loss {best_val_loss:.4f}.")
                    # Restore best weights
                    self.model.load_state_dict(self.best_weights)
                    break
                    
            if verbose > 0:
                if val_loss is not None:
                    self.node.get_logger().info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")
                else:
                    self.node.get_logger().info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f}")

    def call(self, x):
        """Make predictions with the trained model.

        :param x: Input data for prediction. Can be numpy array or torch tensor.
        :type x: np.ndarray or torch.Tensor
        :return: Model predictions as a numpy array, or None if model is not configured.
        :rtype: np.ndarray or None
        """        
        if not self.configured:
            return None
            
        # Convert to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
            
        if len(x.shape) == 1:
            x = x.unsqueeze(1)
            
        x = x.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(x)
            
        return predictions.cpu().numpy()

    def evaluate(self, x_test, y_test):
        """Evaluate the model on test data and return Mean Absolute Error (MAE).

        :param x_test: Test input data (features).
        :type x_test: np.ndarray or torch.Tensor
        :param y_test: Test target data (labels).
        :type y_test: np.ndarray or torch.Tensor
        :return: Mean Absolute Error between predictions and true values, or None if model is not configured.
        :rtype: float or None
        """        
        if not self.configured:
            return None
            
        # Convert to tensors
        if isinstance(x_test, np.ndarray):
            x_test = torch.FloatTensor(x_test)
        if isinstance(y_test, np.ndarray):
            y_test = torch.FloatTensor(y_test)
            
        if len(x_test.shape) == 1:
            x_test = x_test.unsqueeze(1)
        if len(y_test.shape) == 1:
            y_test = y_test.unsqueeze(1)
            
        x_test = x_test.to(self.device)
        y_test = y_test.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(x_test)
            mae = torch.mean(torch.abs(predictions - y_test))
            
        return mae.cpu().item()

    def get_weights(self):
        """Get the current model , uses the state dictionary."""
        if not self.configured:
            self.node.get_logger().warning("Model not configured. Cannot get weights.")
            return None
        return self.model.state_dict()

    def set_weights(self, weights):
        """Set the model weights from a state dictionary.

        :param weights: PyTorch state dictionary containing model weights and biases.
        :type weights: dict
        :return: True if weights were successfully loaded, False otherwise.
        :rtype: bool
        """        
        if not self.configured:
            self.node.get_logger().warning("Model not configured. Cannot set weights.")
            return False
            
        try:
            self.model.load_state_dict(weights)
            self.node.get_logger().info("Weights set successfully.")
            return True
        except Exception as e:
            self.node.get_logger().error(f"Failed to set weights: {e}")
            return False


class ANNModel(nn.Module):
    """PyTorch neural network model."""
    
    def __init__(self, input_size, output_size, hidden_layers=[128], 
                 hidden_activation='relu', output_activation='sigmoid'):
        """Initialize the PyTorch neural network model with configurable architecture.

        :param input_size: Number of input features to the network.
        :type input_size: int
        :param output_size: Number of output features from the network.
        :type output_size: int
        :param hidden_layers: List of integers specifying the size of each hidden layer, defaults to [128]
        :type hidden_layers: list, optional
        :param hidden_activation: Activation function for hidden layers ('relu', 'tanh', 'sigmoid'), defaults to 'relu'
        :type hidden_activation: str, optional
        :param output_activation: Activation function for output layer ('sigmoid', 'tanh', 'relu', 'linear'), defaults to 'sigmoid'
        :type output_activation: str, optional
        """        
        super(ANNModel, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # Input layer
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_layers:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.layers.append(nn.LayerNorm(hidden_size))
            self.layers.append(self._get_activation(hidden_activation))
            self.layers.append(nn.Dropout(0.1))
            prev_size = hidden_size
            
        # Output layer
        self.layers.append(nn.Linear(prev_size, output_size))
        
        # Output activation
        if output_activation == 'sigmoid':
            self.layers.append(nn.Sigmoid())
        elif output_activation == 'tanh':
            self.layers.append(nn.Tanh())
        elif output_activation == 'relu':
            self.layers.append(nn.ReLU())
        # No activation for linear output
        
    def _get_activation(self, activation_name):
        """Get activation function by name.

        :param activation_name: String name of the activation function.
        :type activation_name: str
        :return: Activation function module.
        :rtype: nn.Module
        """
        if activation_name == 'relu':
            return nn.ReLU()
        elif activation_name == 'tanh':
            return nn.Tanh()
        elif activation_name == 'sigmoid':
            return nn.Sigmoid()
        else:
            return nn.ReLU()  # Default
            
    def forward(self, x):
        """Forward pass through the network.

        :param x: Input tensor.
        :type x: torch.Tensor
        :return: Output tensor after passing through the network.
        :rtype: torch.Tensor
        """        """"""
        for layer in self.layers:
            x = layer(x)
        return x

class AsymmetricMSELoss(nn.Module):
    """Asymmetric Mean Squared Error Loss that penalizes underestimations and overestimations differently."""
    
    def __init__(self, underestimation_penalty=1.0, overestimation_penalty=1.0):
        """Initialize the AsymmetricMSELoss with different penalties for underestimation and overestimation.

        :param underestimation_penalty: Penalty factor for underestimations, defaults to 1.0
        :type underestimation_penalty: float, optional
        :param overestimation_penalty: Penalty factor for overestimations, defaults to 1.0
        :type overestimation_penalty: float, optional
        """        
        super(AsymmetricMSELoss, self).__init__()
        self.underestimation_penalty = underestimation_penalty
        self.overestimation_penalty = overestimation_penalty
        
    def forward(self, y_pred, y_true):
        """Forward pass to compute the asymmetric mean squared error loss.

        :param y_pred: Predicted values.
        :type y_pred: torch.Tensor
        :param y_true: True target values.
        :type y_true: torch.Tensor
        :return: Computed asymmetric mean squared error loss.
        :rtype: torch.Tensor
        """        
        error = y_pred - y_true
        weight = torch.where(error < 0, self.underestimation_penalty, self.overestimation_penalty)
        return torch.mean(weight * torch.square(error))

class Evaluator:
    """
    Class that evaluates the success rate of a model based on its predictions.
    """    
    def __init__(self, node:CognitiveNode, learner:Learner,  buffer:EpisodicBuffer, **params) -> None:
        """
        Constructor of the Evaluator class.

        :param node: Cognitive node that uses this model.
        :type node: CognitiveNode
        :param learner: Learner instance to use for predictions.
        :type learner: Learner
        :param buffer: Episodic buffer to use.
        :type buffer: cognitive_nodes.episodic_buffer.EpisodicBuffer
        """
        self.node = node
        self.learner = learner
        self.buffer = buffer

    def evaluate(self):
        """
        Placeholder method for evaluating the model's success rate.

        :raises NotImplementedError: Not implemented yet.
        """
        raise NotImplementedError


