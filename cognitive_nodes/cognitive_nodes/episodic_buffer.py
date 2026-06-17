import threading
import pandas as pd
import numpy as np
import xarray as xr
from collections import deque
from copy import deepcopy

from core.cognitive_node import CognitiveNode
from core.container import Container, MultiContainer
from cognitive_nodes.episode import Episode, container_to_episode_obj




class EpisodicBuffer:
    """
    Class that creates a buffer of episodes to be used as a STM and learn models. 
    It supports a main buffer for training and a secondary buffer for testing.
    """    
    def __init__(self, node:CognitiveNode, main_size, secondary_size, train_split=1.0, inputs=[], outputs=[], random_seed=0, **params) -> None:
        """Initialize an EpisodicBuffer.

        Creates bounded buffers for episodes and configures label extraction and RNG.

        :param node: Owning cognitive node used for logging and ROS2 integration.
        :type node: CognitiveNode
        :param main_size: Maximum number of episodes stored in the main (training) buffer.
        :type main_size: int
        :param secondary_size: Maximum number of episodes stored in the secondary (testing) buffer.
        :type secondary_size: int
        :param train_split: Fraction in [0, 1] of incoming episodes routed to the main buffer; the remainder goes to secondary.
        :type train_split: float
        :param inputs: Episode fields considered inputs/features (e.g., ['old_perception', 'action']).
        :type inputs: list[str]
        :param outputs: Episode fields considered outputs/targets (e.g., ['perception']).
        :type outputs: list[str]
        :param random_seed: Seed for the internal Numpy RNG used for shuffling and sampling.
        :type random_seed: int
        :param params: Additional keyword parameters reserved for future use.
        :type params: dict

        :return: None
        :rtype: None

        Notes:
        - Buffers are implemented as bounded deques.
        - Labels are derived from observed episodes based on inputs/outputs.
        """        
        self.node=node
        self.train_split=train_split # Percentage of samples used for training
        self.inputs=inputs #Fields of the episode that are considered inputs (Used for prediction)
        self.outputs=outputs #Fields of the episode that are considered outputs (Predicted), or a post calculated value (e.g. Value)
        self.input_labels=[]
        self.output_labels=[]
        self._main_size=main_size
        self._secondary_size=secondary_size
        self.main_buffer=None # Main buffer, used for training
        self.secondary_buffer=None # Secondary buffer, used for testing
        self.main_dataframe_inputs=None # DataFrame for the main buffer
        self.main_dataframe_outputs=None # DataFrame for the main buffer
        self.secondary_dataframe_inputs=None # DataFrame for the secondary buffer
        self.secondary_dataframe_outputs=None # DataFrame for the secondary buffer
        self.new_sample_count_main=0 # Counter for new samples in the main buffer
        self.new_sample_count_secondary=0 # Counter for new samples in the secondary buffer
        self.rng = np.random.default_rng(random_seed) # Configuration of the random number generator
        self.semaphore = threading.Semaphore() # Semaphore for thread-safe operations on the buffer

    def configure_labels(self, episode: Episode):
        """
        Creates the label list.

        :param episode: Episode object.
        :type episode: cognitive_node_interfaces.msg.Episode
        """
        self.node.get_logger().info(f"Configuring labels for episodic buffer: {episode}")
        self.input_labels.clear()
        self.output_labels.clear()
        self.input_labels = self._extract_labels(self.inputs, episode)
        self.output_labels = self._extract_labels(self.outputs, episode)
        self.main_buffer = Container("MainBuffer", self._main_size, container_type="buffer", labels=self.input_labels + self.output_labels)
        self.secondary_buffer = Container("SecondaryBuffer", self._secondary_size, container_type="buffer", labels=self.input_labels + self.output_labels)
        self.node.get_logger().info(f"Configuration finished - Input labels: {self.input_labels}, Output labels: {self.output_labels}")

    def update_labels(self, episode: Episode):
        """
        Updates the label list based on the given episode.

        :param episode: Episode object.
        :type episode: cognitive_node_interfaces.msg.Episode
        """
        self.node.get_logger().debug(f"Updating labels for episodic buffer: {episode}")
        new_input_labels = self._extract_labels(self.inputs, episode)
        new_output_labels = self._extract_labels(self.outputs, episode)
        # Add new input labels and log warnings
        new_inputs = set(new_input_labels) - set(self.input_labels)
        if new_inputs:
            self.input_labels.extend(new_inputs)
            self.node.get_logger().warning(f"New input labels {new_inputs} added to episodic buffer.")

        # Add new output labels and log warnings
        new_outputs = set(new_output_labels) - set(self.output_labels)
        if new_outputs:
            self.output_labels.extend(new_outputs)
            self.node.get_logger().warning(f"New output labels {new_outputs} added to episodic buffer.")


    def add_episode(self, episode: Episode, reward=0.0):
        """Add an episode to the buffer.

        Validates content, updates labels, and routes the episode to the main or
        secondary buffer based on train_split.

        :param episode: Episode instance to add.
        :type episode: Episode
        :param reward: Optional reward value (unused in EpisodicBuffer; used by TraceBuffer overrides).
        :type reward: float

        :return: None
        :rtype: None

        Notes:
        - Empty episodes (w.r.t configured inputs/outputs) are skipped with a warning.
        - Labels are configured lazily on the first episode and updated when new dimensions appear.
        """
        self.semaphore.acquire()
        if self.empty_episode(episode, self.inputs, self.outputs):
            self.node.get_logger().warning("The episode is empty, not adding to buffer.")
        else:
            if (not self.input_labels and self.inputs) or (not self.output_labels and self.outputs):
                self.configure_labels(episode)
            else:
                self.update_labels(episode)
            if self.rng.uniform() < self.train_split:
                # Add to main buffer
                self.main_buffer.push(episode.obtain_flattened_episode())
                self.new_sample_count_main += 1
            else:
                # Add to secondary buffer
                self.secondary_buffer.push(episode.obtain_flattened_episode())
                self.new_sample_count_secondary += 1
        self.semaphore.release()
        
    def remove_episode(self, index=None, remove_from_main=True):
        """Remove an episode from the buffer.

        If index is provided, removes the episode at that position; otherwise,
        removes the oldest (leftmost) episode from the selected buffer.

        :param index: Position to remove. If None, removes the oldest entry.
        :type index: int | None
        :param remove_from_main: True to remove from main_buffer; False for secondary_buffer.
        :type remove_from_main: bool

        :return: None
        :rtype: None

        """
        self.semaphore.acquire() # Remember to apply the semaphore once the method is implemented to ensure thread safety
        raise NotImplementedError("remove_episode method is not implemented yet.")
        self.semaphore.release()

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
        Checks if the episode is compatible with the current buffer configuration.

        :param episode: Episode object to check compatibility.
        :type episode: cognitive_node_interfaces.msg.Episode
        :return: True if compatible, False otherwise.
        :rtype: bool
        """
        raise NotImplementedError("is_compatible method is not implemented yet.")
        

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
        Method to obtain a sample from the buffer.

        :param index: Index of the sample to obtain.
        :type index: int
        :param main: Whether to get the sample from the main buffer or secondary buffer.
        :type main: bool
        :return: The requested sample.
        :rtype: list
        """
        if main:
            return self.main_buffer.read(index)
        else:
            return self.secondary_buffer.read(index)

    def get_dataset(self, shuffle=False, n_samples=None):
        """
        Returns the dataset as numpy arrays.

        :param shuffle: Option to shuffle the dataset , defaults to False
        :type shuffle: bool, optional
        :param n_samples: Option to limit the number of samples returned, defaults to None
        :type n_samples: int, optional
        :return: Tuple containing training inputs, training outputs, test inputs, and test outputs as numpy arrays.
        :rtype: tuple
        """
        x_train, y_train = self.get_samples_from_buffer(self.main_buffer, shuffle=shuffle, n_samples=n_samples)
        x_test, y_test = self.get_samples_from_buffer(self.secondary_buffer, shuffle=shuffle, n_samples=n_samples)
        return x_train, y_train, x_test, y_test


    def get_train_samples(self, shuffle=False, n_samples=None):
        """Returns the training samples as lists of input and output dicts.

        :param shuffle: Option to shuffle the samples, defaults to False
        :type shuffle: bool, optional
        :param n_samples: Option to limit the number of samples returned, defaults to None
        :type n_samples: int, optional
        :return: Tuple containing training inputs and training outputs as numpy arrays.
        :rtype: tuple
        """
        return self.get_samples_from_buffer(self.main_buffer, shuffle=shuffle, n_samples=n_samples)

    def get_test_samples(self, shuffle=False, n_samples=None):
        """Returns the test samples as lists of input and output dicts.

        :param shuffle: Option to shuffle the samples, defaults to False
        :type shuffle: bool, optional
        :param n_samples: Option to limit the number of samples returned, defaults to None
        :type n_samples: int, optional
        :return: Tuple containing test inputs and test outputs as numpy arrays.
        :rtype: tuple
        """
        return self.get_samples_from_buffer(self.secondary_buffer, shuffle=shuffle, n_samples=n_samples)
    
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
    def main_max_size(self):
        """
        Returns the max size of the main buffer.

        :return: The maximum size of the main buffer.
        :rtype: int
        """        
        if self.main_buffer is None:
            return self.main_size
        return self.main_buffer.max_size if self.main_buffer.max_size is not None else float('inf')

    @property
    def secondary_max_size(self):
        """
        Returns the max size of the secondary buffer.

        :return: The maximum size of the secondary buffer.
        :rtype: int
        """        
        if self.secondary_buffer is None:
            return self.secondary_size
        return self.secondary_buffer.max_size if self.secondary_buffer.max_size is not None else float('inf')

    @property
    def main_size(self):
        """
        Returns the current size of the main buffer.

        :return: The current size of the main buffer.
        :rtype: int
        """
        return len(self.main_buffer) if self.main_buffer is not None else 0

    @property
    def secondary_size(self):
        """
        Returns the current size of the secondary buffer.

        :return: The current size of the secondary buffer.
        :rtype: int
        """        
        return len(self.secondary_buffer) if self.secondary_buffer is not None else 0
    
    # HELPER METHODS
    @staticmethod
    def empty_episode(episode: Episode, inputs: list, outputs: list):
        """
        Checks if an episode is empty.

        :param episode: Episode object.
        :type episode: Episode
        :param inputs: Input labels to check.
        :type inputs: list
        :param outputs: Output labels to check.
        :type outputs: list
        :return: True if the episode is empty, False otherwise.
        :rtype: bool
        """
        empty = True
        instances = inputs + outputs
        for instance in instances:
            if getattr(episode, instance):
                empty = False
        return empty

    @staticmethod
    def buffer_to_dataframe(buffer: Container, labels: list[str], ordered: bool = True, include_timestamp: bool = False) -> pd.DataFrame:
        # Empty/None buffer -> return typed empty frame with expected columns
        base_cols = list(labels)
        cols = (["timestamp"] + base_cols) if include_timestamp else base_cols
        if buffer is None or len(buffer) == 0:
            return pd.DataFrame(columns=cols)

        # Validate requested labels against container feature labels
        feature_labels = set(buffer.feature_labels)
        missing = [lab for lab in labels if lab not in feature_labels]
        if missing:
            raise ValueError(f"Labels not present in buffer: {missing}")

        # Read valid samples and project selected feature columns
        arr = buffer.read(None, ordered=ordered).sel(features=labels)

        df = pd.DataFrame(arr.values, columns=labels)

        if include_timestamp:
            ts = np.asarray(arr.coords["timestamp"].values, dtype=np.float64)
            df.insert(0, "timestamp", ts)

        return df

    def get_samples_from_buffer(self, buffer: Container, shuffle=False, n_samples=None):
        """
        Internal helper to get (inputs, outputs) numpy arrays from a buffer.

        :param buffer: Episode buffer to extract samples from.
        :type buffer: Container
        :param shuffle: Whether to shuffle the samples, defaults to False
        :type shuffle: bool, optional
        :param n_samples: Number of samples to extract, defaults to None
        :type n_samples: int, optional
        :return: Tuple of (inputs, outputs) numpy arrays
        :rtype: tuple
        """
        data= buffer.read(None, ordered=True)
        inputs = data.sel(features=self.input_labels).values if self.input_labels else np.empty((0, 0)) 
        outputs = data.sel(features=self.output_labels).values if self.output_labels else np.empty((0, 0))
        if shuffle:
            inputs, outputs = self._shuffle_dataset(inputs, outputs)

        # If requested, sample up to n_samples without replacement using the object's RNG
        if n_samples is not None:
            total = inputs.shape[0]
            if total > n_samples:
                idx = self.rng.choice(total, size=n_samples, replace=False)
                inputs = inputs[idx]
                if outputs.size:
                    outputs = outputs[idx]
        return inputs, outputs
    
    def input_episodes_to_matrix(self, episodes: Episode):
        buffer = episodes.obtain_flattened_episode()
        data = buffer.read(ordered=True)
        inputs = data.sel(features=self.input_labels).values if self.input_labels else np.empty((0,0))
        return inputs
    
    def matrix_to_output_episodes(self, matrix, timestamps=None):
        if not timestamps:
            timestamps = np.zeros(matrix.shape[0])
        if not self.output_labels:
            return Episode()
        buffer = self._matrix_to_buffer(matrix, timestamps, self.output_labels)
        episode = container_to_episode_obj(buffer)
        return episode
    
    
    

    def _shuffle_dataset(self, inputs, outputs):
        """
        Helper method to shuffle the dataset represented by numpy arrays.

        :param inputs: Inputs array.
        :type inputs: np.ndarray
        :param outputs: Outputs array.
        :type outputs: np.ndarray
        :return: Tuple of shuffled (inputs, outputs) arrays.
        :rtype: tuple
        """        
        idx = self.rng.permutation(inputs.shape[0])
        inputs = inputs[idx]
        if len(outputs) > 0:
            outputs = outputs[idx]
        return inputs, outputs

    @staticmethod
    def _extract_labels(io_list, episode):
        """
        Helper method to obtain the label list from a given episode.

        :param io_list: List that specifies the fields to extract labels from.
        :type io_list: list
        :param episode: Episode object to extract labels from.
        :type episode: Episode
        :param label_list: List to append the extracted labels to.
        :type label_list: list
        """
        label_list=[]
        for io in io_list:
            if hasattr(episode, io):
                io_obj = getattr(episode, io)
                # Extract feature labels if available (for Container objects)
                io_labels = getattr(io_obj, 'feature_labels', [])
                if not io_labels and isinstance(io_obj, Container):
                    raise ValueError(f"Container object for '{io}' does not have 'feature_labels' attribute or it is empty.")
                label_list.extend([f"{io}:{label}" for label in io_labels])
            else:
                label_list.append(io)  # If it's not an attribute, treat it as a direct label
        return label_list
    
    @staticmethod
    def _matrix_to_buffer(matrix: np.ndarray, timestamps: np.ndarray, labels: list[str], name: str = "result_buffer") -> Container:
        """
        Helper method to convert a numpy array back into a Container buffer.

        :param matrix: Numpy array containing episode data.
        :type matrix: np.ndarray
        :param labels: List of labels corresponding to the columns in the matrix.
        :type labels: list[str]
        :return: Container object populated with the data from the matrix.
        :rtype: Container
        """
        if matrix.shape[1] != len(labels):
            raise ValueError(f"Number of columns in matrix ({matrix.shape[1]}) does not match number of labels ({len(labels)}).")
        
        if matrix.shape[0] != len(timestamps):
            raise ValueError(f"Number of rows in matrix ({matrix.shape[0]}) does not match number of timestamps ({len(timestamps)}).")
        
        buffer = Container(name, max_size=matrix.shape[0], container_type="buffer", labels=labels)
        buffer.push(matrix, src_labels=labels, src_dtype=matrix.dtype, timestamps=timestamps)
        return buffer

class TraceBuffer(EpisodicBuffer):
    """
    Trace Buffer class, a specialized version of the Episodic Buffer that stores traces of episodes.
    """
    def __init__(self, node, main_size, secondary_size=0, max_traces=10, min_traces=1, max_antitraces=5, train_split=1.0, inputs=[], outputs=[], evaluation_method='linear', reward_factor=1.0, random_seed=0, **params):
        """Initialize a TraceBuffer for storing episode traces with utility values.

        Creates buffers for successful traces and antitraces, evaluating episode utilities
        based on final rewards using configurable evaluation methods.

        :param node: Owning cognitive node used for logging and ROS2 integration.
        :type node: CognitiveNode
        :param main_size: Maximum number of episodes stored in a single trace before completion.
        :type main_size: int
        :param secondary_size: Maximum number of episodes in secondary buffer (unused in TraceBuffer), defaults to 0
        :type secondary_size: int, optional
        :param max_traces: Maximum number of successful traces to retain, defaults to 10
        :type max_traces: int, optional
        :param min_traces: Minimum number of traces required before allowing antitraces to be added, defaults to 1
        :type min_traces: int, optional
        :param max_antitraces: Maximum number of failed traces (antitraces) to retain, defaults to 5
        :type max_antitraces: int, optional
        :param train_split: Fraction of episodes for training (inherited but unused in TraceBuffer), defaults to 1.0
        :type train_split: float, optional
        :param inputs: Episode fields considered inputs/features (e.g., ['old_perception', 'action']), defaults to []
        :type inputs: list, optional
        :param outputs: Episode fields considered outputs/targets (e.g., ['perception']), defaults to []
        :type outputs: list, optional
        :param evaluation_method: Method for computing utility values ('linear', 'exponential', 'goal_only'), defaults to 'linear'
        :type evaluation_method: str, optional
        :param reward_factor: Multiplicative factor applied to the final reward in utility calculation, defaults to 1.0
        :type reward_factor: float, optional
        :param random_seed: Seed for the internal Numpy RNG used for shuffling and sampling, defaults to 0
        :type random_seed: int, optional
        :param params: Additional keyword parameters reserved for future use.
        :type params: dict


        Notes:
        - Traces are stored as lists of (episode, utility) tuples.
        - Utility values are computed when a trace completes based on the evaluation method.
        - Antitraces represent failed sequences and have zero utility values.
        """
        super().__init__(node, main_size, secondary_size, train_split, inputs, outputs, random_seed, **params)
        self.max_traces = max_traces
        self.max_antitraces = max_antitraces
        self.traces_buffer = None
        self.antitraces_buffer = None
        self.new_traces = 0
        self.min_utility_fraction = 0.01
        self.min_traces = float(min_traces)
        self.evaluation_method = evaluation_method
        self.reward_factor = reward_factor

    def configure_labels(self, episode: Episode):
        """
        Configures the labels for the trace buffer based on the given episode.

        :param episode: Episode object to configure labels from.
        :type episode: Episode
        """
        super().configure_labels(episode)
        self.traces_buffer = MultiContainer("TracesBuffer", max_traces=self.max_traces, max_size=self.main_max_size, container_type="trace_buffer", labels=self.input_labels + self.output_labels + ["utility"])
        self.antitraces_buffer = MultiContainer("AntitracesBuffer", max_traces=self.max_antitraces, max_size=self.main_max_size, container_type="trace_buffer", labels=self.input_labels + self.output_labels + ["utility"])
        self.node.get_logger().info(f"Configured labels for trace buffers")

    def add_episode(self, episode, reward=0.0):
        """Add an episode to the trace buffer and complete the trace if reward is positive.

        Appends the episode to the main buffer. If a positive reward is provided,
        evaluates utilities for all episodes in the current trace, stores the trace,
        and clears the buffer to begin a new trace.

        :param episode: Episode instance to add to the current trace.
        :type episode: Episode
        :param reward: Reward value for the trace; positive values trigger trace completion, defaults to 0.0
        :type reward: float, optional
        :raises ValueError: If episode is not of type Episode.

        :return: None
        :rtype: None

        Notes:
        - Only episodes of type Episode are accepted.
        - Labels are configured on the first episode if not already set.
        - Positive rewards trigger utility evaluation and trace storage.
        - The buffer is cleared after storing a successful trace.
        """
        self.semaphore.acquire()
        if type(episode) is not Episode:
            raise ValueError("The episode must be of type Episode.")
        if (not self.input_labels and self.inputs) or (not self.output_labels and self.outputs):
            self.configure_labels(episode)
        self.main_buffer.push(episode.obtain_flattened_episode())
        self.new_sample_count_main += 1

        #Add corresponding trace
        if reward > 0: # If the reward is positive, consider it a successful trace
            trace=self.main_buffer # Trace corresponds to the contents of the main buffer
            self._add_trace(trace, reward, clear_main_buffer=True) # Add the trace to the traces buffer with the corresponding reward, and clear the main buffer for the next trace
        self.semaphore.release()

    def add_trace (self, trace: Episode, reward: float = 0, reward_trace: np.ndarray = None):
        """Public method to add a trace to the buffer with an associated reward.

        This method serves as the public interface for adding complete traces along
        with their rewards. It validates the trace format and delegates to the internal
        _add_trace method for processing.
        :param trace: Episodes that belong to the trace.
        :type trace: Episode
        :param reward: Reward value for the trace; positive values indicate success, defaults to 0
        :type reward: float, optional
        :param reward_trace: Optional precomputed utility values for the trace; if None, utilities are computed using the evaluation method, defaults to None
        :type reward_trace: np.ndarray | None, optional
        """
        self.semaphore.acquire()
        if (not self.input_labels and self.inputs) or (not self.output_labels and self.outputs):
            self.configure_labels(trace)

        flat_episodes = trace.obtain_flattened_episode()
        self._add_trace(flat_episodes, reward, reward_trace, clear_main_buffer=False)
        self.semaphore.release()


    def _add_trace(self, trace: Container, reward: float = 0, reward_trace: np.ndarray = None, clear_main_buffer: bool = True):
        """Add a trace directly to the buffer with an associated reward.

        This method allows adding a complete trace (sequence of episodes) along with
        its reward and optional precomputed utility values. The trace is stored in
        the appropriate buffer based on the reward value.

        :param trace: DataArray containing the sequence of episodes to add as a trace.
        :type trace: xarray.DataArray
        :param reward: Reward value for the trace; positive values indicate success, defaults to 0
        :type reward: float, optional
        :param reward_trace: Optional precomputed utility values for the trace; if None, utilities are computed using the evaluation method, defaults to None
        :type reward_trace: np.ndarray | None, optional
        """
        trace_array = trace.read(ordered=True)
        if not reward_trace:
            reward_trace = self.evaluate_trace(reward, n=trace_array.shape[0])
        trace_data = np.concatenate((trace_array.values, reward_trace), axis=1) # Combine episode data with utility values
        trace_labels = trace.feature_labels + ["utility"]
        trace_timestamps = trace_array.coords["timestamp"].values 
        dtype = trace_array.dtype
        self.traces_buffer.push(trace_data, src_labels=trace_labels, src_dtype=dtype, timestamps=trace_timestamps)
        self.new_traces += 1
        self.node.get_logger().info(f"Adding trace with {trace_array.shape[0]} episodes. New traces: {self.new_traces}")
        if clear_main_buffer:
            self.clear()

    def add_antitrace(self):
        """Add an antitrace to the buffer if enough traces exist.
        """        
        self.semaphore.acquire()
        if self.n_traces >= self.min_traces: # If the buffer is full, and there are enough traces, add an antitrace
            self.node.get_logger().info("Adding antitrace")
            trace=self.main_buffer.read(None, ordered=True) # Ensure the buffer is ordered for correct utility assignment
            trace_data = np.concatenate((trace.values, np.zeros((len(trace.values), 1))), axis=1) # Combine episode data with utility values
            trace_labels = self.main_buffer.feature_labels + ["utility"]
            trace_timestamps = trace.coords["timestamp"].values 
            dtype = self.main_buffer.data.dtype
            self.antitraces_buffer.push(trace_data, src_labels=trace_labels, src_dtype=dtype, timestamps=trace_timestamps)
            self.clear()
        self.semaphore.release()

    def evaluate_trace(self, reward, n=None):
        """Compute utility values for each episode in the current trace.

        Uses the configured evaluation method to assign utility values to all episodes
        in the main buffer based on the final reward. The evaluation method determines
        how utility is distributed across the episode sequence.

        :param reward: Final reward value for the trace used to compute utilities.
        :type reward: float
        :param n: Optional number of episodes in the trace; if None, uses the current size of the main buffer, defaults to None
        :type n: int | None, optional
        :return: List of utility values, one per episode in the trace.
        :rtype: np.ndarray
        """
        if n is None:
            n = len(self.main_buffer)
            if n == 0:
                return []
        min_val = reward * self.min_utility_fraction
        values = getattr(self, f"eval_{self.evaluation_method}", self.eval_default)(reward, min_val, self.reward_factor, n, self.main_max_size)
        values = np.array(values).reshape(-1, 1)
        return values
    

    def _get_samples_from_traces(self, traces_buffer: MultiContainer, n_samples=None, shuffle=True):
        """Flatten stored traces into a single buffer with corresponding utilities.

        Optionally samples a subset of traces, then flattens all (episode, utility) pairs
        from the selected traces into a single sequential buffer.

        :param n_samples: Maximum number of traces to sample; if None, uses all traces, defaults to None
        :type n_samples: int | None, optional
        :return: Tuple of (episode buffer, utilities array) containing flattened trace data.
        :rtype: tuple[list, np.ndarray]
        """
        if n_samples is not None:
            if traces_buffer.n_traces > n_samples:
                idx = self.rng.choice(traces_buffer.n_traces, size=n_samples, replace=False)
                flattened_traces = traces_buffer.read_flattened(idx)
            else:
                flattened_traces = traces_buffer.read_flattened(None)
        else:
            flattened_traces = traces_buffer.read_flattened(None)
        utilities = flattened_traces.sel(features="utility").values.reshape(-1)
        buffer = flattened_traces.drop_sel(features=["utility"]).values
        if shuffle:
            buffer, utilities = self._shuffle_dataset(buffer, utilities)
        return buffer, utilities

    def get_dataset(self, shuffle=True, n_samples=None):
        """Return training dataset from flattened traces as numpy arrays.

        Flattens stored traces into episode states and utilities, optionally shuffling
        the resulting dataset.

        :param shuffle: Whether to shuffle the dataset, defaults to True
        :type shuffle: bool, optional
        :param n_samples: Maximum number of traces to include; if None, uses all traces, defaults to None
        :type n_samples: int | None, optional
        :return: Tuple of (states, utilities) numpy arrays for training.
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        x_train, y_train = self._get_samples_from_traces(self.traces_buffer, n_samples=n_samples, shuffle=shuffle)
        return x_train, y_train

    def episodes_to_matrix(self):
        pass
    
    def reset_new_sample_count(self, main=True, secondary=True):
        """
        Resets the new sample count for the trace buffer.

        :param main: Unused in this class, defaults to True.
        :type main: bool
        :param secondary: Unused in this class, defaults to True.
        :type secondary: bool
        """
        self.new_traces = 0

    @staticmethod
    def eval_default(reward, min_val, reward_factor, n, full_length):
        """Default evaluation method placeholder.

        :param reward: Final reward for the trace.
        :type reward: float
        :param min_val: Minimum utility value derived from the reward (e.g., reward * min_utility_fraction).
        :type min_val: float
        :param reward_factor: Multiplicative factor applied to the final utility value.
        :type reward_factor: float
        :param n: Number of episodes in the current trace.
        :type n: int
        :param full_length: Maximum possible length of a trace (main buffer capacity).
        :type full_length: int
        :raises NotImplementedError: Always, as this method is a placeholder.
        """
        raise NotImplementedError(f"Evaluation method requested is not implemented.")

    @staticmethod
    def eval_linear(reward, min_val, reward_factor, n, full_length):
        """Linear evaluation: interpolate utilities from a start value to the final reward.

        Computes a start value consistent with the full-length trace and linearly
        interpolates utilities across episodes, ending at reward * reward_factor.

        :param reward: Final reward for the trace.
        :type reward: float
        :param min_val: Minimum utility value derived from the reward.
        :type min_val: float
        :param reward_factor: Multiplicative factor applied to the final utility value.
        :type reward_factor: float
        :param n: Number of episodes in the current trace.
        :type n: int
        :param full_length: Maximum possible length of a trace (main buffer capacity).
        :type full_length: int
        :return: List of length n with linearly increasing utility values.
        :rtype: list[float]
        """
        if n == 1:
            return [reward]
        
        # Compute the start value at the equivalent position in a full-length trace
        start_val = min_val + (reward - min_val) * (full_length - n) / (full_length - 1)
        
        # Linear interpolation from start_val to reward over n steps
        values = []
        for i in range(n-1):
            value = start_val + (reward - start_val) * i / (n - 1)
            values.append(value)
        values.append(reward*reward_factor)
        return values

    @staticmethod
    def eval_exponential(reward, min_val, reward_factor, n, full_length):
        """Exponential evaluation: grow utilities exponentially towards the final reward.

        Uses k = ln(reward/min_val) / (full_length - 1) and evaluates utilities at
        positions [full_length - n, ..., full_length - 2], appending reward * reward_factor
        as the final value.

        :param reward: Final reward for the trace.
        :type reward: float
        :param min_val: Minimum utility value derived from the reward; adjusted if non-positive.
        :type min_val: float
        :param reward_factor: Multiplicative factor applied to the final utility value.
        :type reward_factor: float
        :param n: Number of episodes in the current trace.
        :type n: int
        :param full_length: Maximum possible length of a trace (main buffer capacity).
        :type full_length: int
        :return: List of length n with exponentially increasing utility values.
        :rtype: list[float]
        """
        if n == 1:
            return [reward*reward_factor]
        
        # Ensure min_val is positive for logarithm calculation
        if min_val <= 0:
            min_val = 0.001
        
        # Calculate the exponential growth rate based on full sequence
        k = np.log(reward / min_val) / (full_length - 1)
        
        # Generate exponential values for the last n positions of the full sequence
        values = []
        start_position = full_length - n
        for i in range(n-1):
            position = start_position + i
            value = min_val * np.exp(k * position)
            values.append(value)
        values.append(reward*reward_factor)
        return values
    
    @staticmethod
    def eval_goal_only(reward, min_val, reward_factor, n, full_length):
        """Goal-only evaluation: zero utility except at the final episode.

        Assigns 0.0 to all episodes except the last, which receives reward * reward_factor.

        :param reward: Final reward for the trace.
        :type reward: float
        :param min_val: Minimum utility value (unused in this method).
        :type min_val: float
        :param reward_factor: Multiplicative factor applied to the final utility value.
        :type reward_factor: float
        :param n: Number of episodes in the current trace.
        :type n: int
        :param full_length: Maximum possible length of a trace (unused in this method).
        :type full_length: int
        :return: List of length n with zero utilities except the final element.
        :rtype: list[float]
        """
        values = [0.0] * (n - 1) + [reward*reward_factor]
        return values

    @property
    def n_traces(self):
        """Current number of stored traces.

        :return: Count of traces in the buffer.
        :rtype: int
        """
        return self.traces_buffer.n_traces if self.traces_buffer is not None else 0

    @property
    def n_antitraces(self):
        """Current number of stored antitraces.

        :return: Count of antitraces in the buffer.
        :rtype: int
        """
        return self.antitraces_buffer.n_traces if self.antitraces_buffer is not None else 0

