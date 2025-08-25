import rclpy
import pandas as pd
import numpy as np
from collections import deque
from copy import deepcopy

from core.cognitive_node import CognitiveNode
from cognitive_nodes.episode import Episode, Action, episode_msg_to_obj

from cognitive_node_interfaces.msg import Episode as EpisodeMsg




class EpisodicBuffer:
    """
    Class that creates a buffer of episodes to be used as a STM and learn models. 

    WORK IN PROGRESS
    """    
    def __init__(self, node:CognitiveNode, main_size, secondary_size, train_split=0.8, inputs=[], outputs=[], random_seed=0, **params) -> None:
        """
        Constructor of the EpisodicBuffer class.

        :param node: Cognitive node that will contain this episodic buffer.
        :type node: core.cognitive_node.CognitiveNode
        :param episode_topic: Topic where episodes are read.
        :type episode_topic: str
        :param episode_msg: Message type of the episodes topic.
        :type episode_msg: str
        :param max_size: Maximum size of the episodic buffer, defaults to 500.
        :type max_size: int
        :param inputs: List to configure inputs (from the attributes of the episode message) considered in the buffer, defaults to [].
        :type inputs: list
        :param outputs: List to configure outputs (from the attributes of the episode message) considered in the buffer, defaults to [].
        :type outputs: list
        """        
        self.node=node
        self.train_split=train_split # Percentage of samples used for training
        self.inputs=inputs #Fields of the episode that are considered inputs (Used for prediction)
        self.outputs=outputs #Fields of the episode that are considered outputs (Predicted), or a post calculated value (e.g. Value)
        self.input_labels=[]
        self.output_labels=[]
        self.is_input=[]
        self.main_buffer=deque(maxlen=main_size) # Main buffer, used for training
        self.secondary_buffer=deque(maxlen=secondary_size) # Secondary buffer, used for testing
        self.main_dataframe_inputs=None # DataFrame for the main buffer
        self.main_dataframe_outputs=None # DataFrame for the main buffer
        self.secondary_dataframe_inputs=None # DataFrame for the secondary buffer
        self.secondary_dataframe_outputs=None # DataFrame for the secondary buffer
        self.new_sample_count_main=0 # Counter for new samples in the main buffer
        self.new_sample_count_secondary=0 # Counter for new samples in the secondary buffer
        self.rng = np.random.default_rng(random_seed) # Configuration of the random number generator

    def configure_labels(self, episode: Episode):
        """
        Creates the label list.

        :param episode: Episode object.
        :type episode: cognitive_node_interfaces.msg.Episode
        """
        self.node.get_logger().info(f"Configuring labels for episodic buffer: {episode}")
        self.input_labels.clear()
        self.output_labels.clear()
        self._extract_labels(self.inputs, episode, self.input_labels)
        self._extract_labels(self.outputs, episode, self.output_labels)
        self.node.get_logger().info(f"Configuration finished - Input labels: {self.input_labels}, Output labels: {self.output_labels}")

    def add_episode(self, episode: Episode):
        self.node.get_logger().info(f"DEBUG - Input labels: {self.input_labels}, Output labels: {self.output_labels} / Inputs: {self.inputs}, Outputs: {self.outputs}")
        if (not self.input_labels and self.inputs) or (not self.output_labels and self.outputs):
            self.configure_labels(episode)
        if self.rng.uniform() < self.train_split:
            # Add to main buffer
            self.main_buffer.append(deepcopy(episode))
            self.new_sample_count_main += 1
        else:
            # Add to secondary buffer
            self.secondary_buffer.append(deepcopy(episode))
            self.new_sample_count_secondary += 1
        
    def remove_episode(self, index=None, remove_from_main=True):
        if remove_from_main:
            if index is not None:
                self.main_buffer.remove(self.main_buffer[index]) 
            else:
                self.main_buffer.popleft()
        else:
            if index is not None:
                self.secondary_buffer.remove(self.secondary_buffer[index])
            else:
                self.secondary_buffer.popleft()

    def clear(self):
        """
        Clears the episodic buffer.
        """
        self.main_buffer.clear()
        self.secondary_buffer.clear()
        self.main_dataframe = None
        self.secondary_dataframe = None
        self.new_sample_count_main = 0
        self.new_sample_count_secondary = 0
    
    def create_dataframes(self):
        """
        Creates pandas DataFrames from the main and secondary buffers.
        """
        if self.input_labels:
            if len(self.main_buffer) > 0:
                self.main_dataframe_inputs = self.buffer_to_dataframe(self.main_buffer, self.input_labels)
            if len(self.secondary_buffer) > 0:
                self.secondary_dataframe_inputs = self.buffer_to_dataframe(self.secondary_buffer, self.input_labels)
        if self.output_labels:
            if len(self.main_buffer) > 0:
                self.main_dataframe_outputs = self.buffer_to_dataframe(self.main_buffer, self.output_labels)
            if len(self.secondary_buffer) > 0:
                self.secondary_dataframe_outputs = self.buffer_to_dataframe(self.secondary_buffer, self.output_labels)

    def is_compatible(self, episode: Episode):
        """
        TODO: CHECK THIS METHOD
        Checks if the episode is compatible with the current buffer configuration.

        :param episode: Episode object to check compatibility.
        :type episode: cognitive_node_interfaces.msg.Episode
        :return: True if compatible, False otherwise.
        :rtype: bool
        """
        if not self.input_labels or not self.output_labels:
            self.configure_labels(episode)
        
        for label in self.input_labels + self.output_labels:
            if label not in episode.__dict__:
                return False
        return True
        

    #### GETTERS / SETTERS ####

    def get_input_labels(self):
        """
        Returns the input labels of the episodic buffer.

        :return: List of input labels.
        :rtype: list
        """
        return self.input_labels
    
    def get_output_labels(self):
        """
        Returns the output labels of the episodic buffer.

        :return: List of output labels.
        :rtype: list
        """
        return self.output_labels

    def get_sample(self, index, main=True):
        """
        WORK IN PROGRESS: Method to obtain a sample from the buffer.

        :param index: Index of the sample to obtain.
        :type index: int
        :param main: Whether to get the sample from the main buffer or secondary buffer.
        :type main: bool
        :return: The requested sample.
        :rtype: list
        """
        if main:
            return self.main_buffer[index]
        else:
            return self.secondary_buffer[index]

    def get_dataset(self, shuffle=False):
        """
        Returns the dataset as numpy arrays.
        If shuffle=True, shuffles the arrays before returning.
        """
        x_train, y_train = self._get_samples_from_buffer(self.main_buffer, shuffle=shuffle)
        x_test, y_test = self._get_samples_from_buffer(self.secondary_buffer, shuffle=shuffle)
        return x_train, y_train, x_test, y_test


    def get_train_samples(self, shuffle=False):
        """
        Returns the training samples as lists of input and output dicts.
        :return: (inputs, outputs) where each is a numpy array
        If shuffle=True, shuffles the arrays before returning.
        """
        return self._get_samples_from_buffer(self.main_buffer, shuffle=shuffle)

    def get_test_samples(self, shuffle=False):
        """
        Returns the test samples as lists of input and output dicts.
        :return: (inputs, outputs) where each is a numpy array
        If shuffle=True, shuffles the arrays before returning.
        """
        return self._get_samples_from_buffer(self.secondary_buffer, shuffle=shuffle)
    
    def get_dataframes(self):
        """
        Returns the DataFrames of the main and secondary buffers.

        :return: Tuple with the main and secondary DataFrames.
        :rtype: tuple
        """
        self.create_dataframes()
        return self.main_dataframe_inputs, self.main_dataframe_outputs, self.secondary_dataframe_inputs, self.secondary_dataframe_outputs
    
    def reset_new_sample_count(self, main=True, secondary=True):
        """
        Resets the new sample count for the main and/or secondary buffers.

        :param main: Whether to reset the main buffer count, defaults to True.
        :type main: bool
        :param secondary: Whether to reset the secondary buffer count, defaults to True.
        :type secondary: bool
        """
        if main:
            self.new_sample_count_main = 0
        if secondary:
            self.new_sample_count_secondary = 0

    @property
    def main_size(self):
        """Returns the max size of the main buffer."""
        return self.main_buffer.maxlen

    @property
    def secondary_size(self):
        """Returns the max size of the secondary buffer."""
        return self.secondary_buffer.maxlen
    
    # HELPER METHODS

    @staticmethod
    def episode_to_flat_dict(episode: Episode, labels):
        """
        Converts an episode to a dict representation matching the labels.
        """
        vector = {}
        dimensions = [label.split(':') for label in labels]
        for label, instance in zip(labels, dimensions):
            if instance[0] == "action":
                if instance[1] == 'policy':
                    value = episode.action.policy_id
                else:
                    value = episode.action.actuation[instance[1]][0][instance[2]]
            elif instance[0] == "reward_list":
                value = episode.reward_list[instance[1]]
            else:
                value = getattr(episode, instance[0])[instance[1]][0][instance[2]]
            vector[label] = value
        return vector
    
    @staticmethod
    def episode_to_vector(episode: Episode, labels):
        """
        Converts an episode to a vector representation matching the labels.
        """
        flat_dict = EpisodicBuffer.episode_to_flat_dict(episode, labels)
        vector = np.zeros(len(labels))
        for i, label in enumerate(labels):
            vector[i] = flat_dict[label]
        return vector
    
    @staticmethod
    def vector_to_episode(vector, labels):
        """
        Converts a vector representation to an episode object.
        """
        episode = Episode()
        if len(labels) != len(vector):
            raise ValueError("The length of the vector does not match the number of labels.")
        for i, label in enumerate(labels):
            instance = label.split(':')
            if instance[0] == "action":
                if instance[1] == 'policy':
                    episode.action.policy_id = vector[i]
                else:
                    if not episode.action.actuation.get(instance[1], None):
                        episode.action.actuation[instance[1]] = [{}]
                    episode.action.actuation[instance[1]][0][instance[2]] = vector[i]
            elif instance[0] == "reward_list":
                episode.reward_list[instance[1]] = vector[i]
            else:
                if not getattr(episode, instance[0]).get(instance[1], None):
                    getattr(episode, instance[0])[instance[1]] = [{}]
                getattr(episode, instance[0])[instance[1]][0][instance[2]] = vector[i]
        return episode

    @staticmethod
    def buffer_to_dict_list(buffer, labels):
        """
        Converts a buffer of episodes to a list of dicts using the given labels.
        """
        return [EpisodicBuffer.episode_to_flat_dict(ep, labels) for ep in buffer]

    @staticmethod
    def buffer_to_dataframe(buffer, labels):
        """
        Converts a buffer of episodes to a pandas DataFrame using the given labels.
        
        :param buffer: Buffer of episodes to convert.
        :type buffer: deque
        :param labels: Labels to use for the DataFrame columns.
        :type labels: list
        :return: DataFrame containing the episodes in the buffer.
        :rtype: pd.DataFrame
        """
        return pd.DataFrame(EpisodicBuffer.buffer_to_dict_list(buffer, labels), columns=labels)
    
    @staticmethod
    def buffer_to_matrix(buffer, labels):
        """
        Converts a buffer of episodes to a numpy matrix using the given labels.
        
        :param buffer: Buffer of episodes to convert.
        :type buffer: deque
        :param labels: Labels to use for the matrix columns.
        :type labels: list
        :return: Numpy matrix containing the episodes in the buffer.
        :rtype: np.ndarray
        """
        return np.array([EpisodicBuffer.episode_to_vector(ep, labels) for ep in buffer])
    
    @staticmethod
    def matrix_to_buffer(matrix, labels):
        """
        Converts a numpy matrix to a buffer of episodes using the given labels.
        
        :param matrix: Numpy matrix to convert.
        :type matrix: np.ndarray
        :param labels: Labels to use for the episodes.
        :type labels: list
        :return: Buffer of episodes created from the matrix.
        :rtype: list
        """
        buffer = [EpisodicBuffer.vector_to_episode(row, labels) for row in matrix]
        return buffer

    def _get_samples_from_buffer(self, buffer, shuffle=False):
        """
        Internal helper to get (inputs, outputs) numpy arrays from a buffer.
        If shuffle=True, shuffles the arrays before returning.
        """
        inputs = self.buffer_to_matrix(buffer, self.input_labels)
        if self.output_labels:
            outputs = self.buffer_to_matrix(buffer, self.output_labels)
        else:
            outputs = []
        if shuffle:
            inputs, outputs = self._shuffle_dataset(inputs, outputs)
        return inputs, outputs

    def _shuffle_dataset(self, inputs, outputs):
        idx = self.rng.permutation(inputs.shape[0])
        inputs = inputs[idx]
        if len(outputs) > 0:
            outputs = outputs[idx]
        return inputs, outputs

    @staticmethod
    def _extract_labels(io_list, episode, label_list):
        for io in io_list:
            io_dict = getattr(episode, io)
            if isinstance(io_dict, Action):
                for group, dims_list in io_dict.actuation.items():
                    dims = dims_list[0]
                    for dim in dims:
                        label_list.append(f"{io}:{group}:{dim}")
                label_list.append(f"{io}:policy:id")
            else:
                for group, dims_list in io_dict.items():
                    if isinstance(dims_list, list):
                        dims = dims_list[0]
                        for dim in dims:
                            label_list.append(f"{io}:{group}:{dim}")
                    else:
                        label_list.append(f"{io}:{group}")

class TraceBuffer(EpisodicBuffer):
    """
    Trace Buffer class, a specialized version of the Episodic Buffer that stores traces of episodes.
    """
    def __init__(self, node, main_size, secondary_size=0, max_traces=10, min_traces=1, max_antitraces=5, train_split=0.8, inputs=[], outputs=[], random_seed=0, **params):
        super().__init__(node, main_size, secondary_size, train_split, inputs, outputs, random_seed, **params)
        self.traces_buffer = deque(maxlen=max_traces)
        self.antitraces_buffer = deque(maxlen=max_antitraces)
        self.new_traces = 0
        self.min_utility_fraction = 0.01
        self.min_traces = min_traces

    def add_episode(self, episode, reward):
        if type(episode) is not Episode:
            raise ValueError("The episode must be of type Episode.")
        if (not self.input_labels and self.inputs) or (not self.output_labels and self.outputs):
            self.configure_labels(episode)
        self.main_buffer.append(deepcopy(episode))
        self.new_sample_count_main += 1

        #Add corresponding trace
        if reward > 0: # If the reward is positive, consider it a successful trace
            utility_trace = self.evaluate_trace(reward)
            self.traces_buffer.append(list(zip(self.main_buffer, utility_trace)))
            self.new_traces += 1
            self.clear()
        elif self.new_sample_count_main == self.main_size and self.n_traces > self.min_traces: # If the buffer is full, and there are enough traces, add an antitrace
            self.antitraces_buffer.append(list(zip(self.main_buffer, np.zeros(self.main_size))))
            self.new_traces += 1
            self.clear()
        elif self.new_sample_count_main == self.main_size: # If the buffer is full but not enough traces, just clear
            self.clear()

    def evaluate_trace(self, reward):
        n = len(self.main_buffer)
        if n == 0:
            return []
        min_val = reward * self.min_utility_fraction
        full_length = self.main_size
        if full_length == 1 or n == 1:
            return [reward]
        # Compute the value at the position of the last element in a full-length trace
        start_val = min_val + (reward - min_val) * (full_length - n) / (full_length - 1)
        # Interpolate from start_val to reward over n steps
        values = [start_val + (reward - start_val) * i / (n - 1) for i in range(n)]
        return values
    
    def get_dataset(self, shuffle=True):
        all_traces = self.traces_buffer + self.antitraces_buffer
        flattened_traces = [item for trace in all_traces for item in trace]
        buffer, utilities = zip(*flattened_traces) if flattened_traces else ([], [])
        utilities = np.array(utilities)
        states, _ = self._get_samples_from_buffer(buffer, shuffle=False)
        x_train, y_train = self._shuffle_dataset(states, utilities) if shuffle else (states, utilities)
        return x_train, y_train
    
    def reset_new_sample_count(self, main=True, secondary=True):
            """
            Resets the new sample count for the trace buffer.

            :param main: Unused in this class, defaults to True.
            :type main: bool
            :param secondary: Unused in this class, defaults to True.
            :type secondary: bool
            """
            self.new_traces = 0

    @property
    def max_traces(self):
        return self.traces_buffer.maxlen

    @property
    def max_antitraces(self):
        return self.antitraces_buffer.maxlen
    
    @property
    def n_traces(self):
        return len(self.traces_buffer)

    @property
    def n_antitraces(self):
        return len(self.antitraces_buffer)


class TestEpisodicBuffer(CognitiveNode):
    """
    Test Episodic Buffer class, this class is a test implementation of the Episodic Buffer.
    It is used to test the functionality of the Episodic Buffer.
    """
    def __init__(self, name='test_episodic_buffer', **params):
        """
        Constructor of the Test Episodic Buffer class.

        :param name: The name of the Test Episodic Buffer instance.
        :type name: str
        """
        super().__init__(name, **params)
        self.episodic_buffer = EpisodicBuffer(self, main_size=10, secondary_size=5, train_split=0.8, inputs=['old_perception', 'action'], outputs=['perception'], random_seed=42)
        self.episode_subscription = self.create_subscription(
            EpisodeMsg,
            '/main_loop/episodes',
            self.episode_callback,
            1
        )

    def episode_callback(self, msg: EpisodeMsg):
        """
        Callback for the episode subscription. It receives an episode message and adds it to the episodic buffer.

        :param msg: The episode message received.
        :type msg: cognitive_node_interfaces.msg.Episode
        """
        episode = episode_msg_to_obj(msg)
        self.episodic_buffer.add_episode(episode)
        self.get_logger().info(f"Episode added to buffer: \n {episode} \n New main samples: {self.episodic_buffer.new_sample_count_main}, New secondary samples: {self.episodic_buffer.new_sample_count_secondary}")

        self.get_logger().info(f"MAIN BUFFER CONTENTS: ")
        for i, episode in enumerate(self.episodic_buffer.main_buffer):
            self.get_logger().info(f" - Episode {i}:\n {episode}")

        self.get_logger().info(f"SECONDARY BUFFER CONTENTS: ")
        for i, episode in enumerate(self.episodic_buffer.secondary_buffer):
            self.get_logger().info(f" - Episode {i}:\n {episode}")
        
        if self.episodic_buffer.new_sample_count_main >= 10:
            x_train, y_train, x_test, y_test = self.episodic_buffer.get_dataset()
            self.get_logger().info(f"Features Train: \n {x_train}")
            self.get_logger().info(f"Targets Train: \n {y_train}")
            self.get_logger().info(f"Features Test: \n {x_test}")
            self.get_logger().info(f"Targets Test: \n {y_test}")

def test_episodic_buffer(args=None):
    rclpy.init(args=args)

    generic_model = TestEpisodicBuffer()

    rclpy.spin(generic_model)

    generic_model.destroy_node()
    rclpy.shutdown()
