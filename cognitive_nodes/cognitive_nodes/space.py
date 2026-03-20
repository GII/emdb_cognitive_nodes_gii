from math import isclose
import numpy
from numpy.lib.recfunctions import structured_to_unstructured, require_fields
import pandas as pd
from sklearn import svm
import tensorflow as tf
from rclpy.node import Node
from rclpy.logging import get_logger

from core.utils import separate_perceptions


class Space(object):
    """A n-dimensional state space."""

    def __init__(self, ident=None, **kwargs):
        """Init attributes when a new object is created.

        :param ident: The name of the space.
        :type ident: str
        """
        self.ident = ident
        self.parent_space = None
        self.logger = get_logger("space_" + str(ident))
        self.logger.info(f"CREATING SPACE: {ident}")


class PointBasedSpace(Space):
    """A state space based on points."""

    def __init__(self, size=15000, **kwargs):
        """
        Init attributes when a new object is created.

        :param size: Maximum number of points that the space can contain, defaults to 5000.
        :type size: int
        """
        self.real_size = size
        self.size = 0
        # These lists must be empty, not None, in order loops correctly operate with empty spaces.
        self.members = []
        self.memberships = []
        super().__init__(**kwargs)

    def populate_space(self, labels, members, memberships):
        """
        Populate the structured array and memberships list based on the given parameters.

        :param point: A perception dictionary describing the structure of the space.
        :type point: dict
        :param members: A flattened list of data with size n_dims * n_data.
        :type members: list
        :param memberships: A list of membership data with size n_data.
        :type memberships: list
        :raises ValueError: If the size of memberships does not match the calculated size of the space.
        """
        if self.size != 0:
            raise RuntimeError("Only an empty space can be populated.")

        # Ensure the size matches the expected dimensions
        if len(memberships) != self.real_size:
            raise ValueError("Size of memberships does not match the space's real size.")

        # Create a structured array using the provided point structure
        point=self.create_point_from_labels(labels)
        self.members = self.create_structured_array(point, None, len(memberships))
        
        # Populate the structured array with the members' data
        n_dims = len(self.members.dtype.names)
        n_data = len(members) // n_dims

        if n_data != len(memberships):
            raise ValueError("Mismatch between members and memberships size.")

        for i in range(n_data):
            member_data = members[i * n_dims:(i + 1) * n_dims]
            self.members[i] = tuple(member_data)

        # Assign memberships
        self.memberships = numpy.array(memberships)
        self.size = len(memberships)
    
    #TODO This method assumes that there is only one element per sensor. See configure_labels in goal.py
    @staticmethod
    def create_point_from_labels(labels):
        """
        Generates a point from a list of labels.

        :param labels: List of labels of the space.
        :type labels: list
        :return: Space point.
        :rtype: dict
        """        
        point={}
        for label in labels:
            elements=label.split("-")
            sensor=elements[1]
            attribute=elements[2]
            if not point.get(sensor):
                point[sensor]=[{attribute: 0.0}]
            else:
                point[sensor][0][attribute]=0.0
        point = separate_perceptions(point)[0]
        print(f"Point: {point}")
        return point

    def create_structured_array(self, perception, base_dtype, size):
        """
        Create a structured array to store points.

        The key is what fields to use. There are three cases:
        - If base_dtype is specified, use the fields in perception that are also in base_dtype.
        - Otherwise, if this space is a specialization, use the fields in perception that are NOT in parent_space.
        - Otherwise, use every field in perception.

        :param perception: The perception that sizes the structured array.
        :type perception: dict
        :param base_dtype: The dtype of the structured array.
        :type base_dtype: numpy.dtype
        :param size: The size of the structured array.
        :type size: int
        :return: The structured array, filled with zeros.
        :rtype: numpy.ndarray
        """
        if getattr(perception, "dtype", None):
            if base_dtype:
                types = [
                    (name, float) for name in perception.dtype.names if name in base_dtype.names
                ]
            elif self.parent_space:
                types = [
                    (name, float)
                    for name in perception.dtype.names
                    if name not in self.parent_space.members.dtype.names
                ]
            else:
                types = perception.dtype
        else:
            if base_dtype:
                types = [
                    (sensor + "_" + attribute, float)
                    for sensor, attributes in perception.items()
                    for attribute in attributes
                    if sensor + "_" + attribute in base_dtype.names
                ]
            elif self.parent_space:
                types = [
                    (sensor + "_" + attribute, float)
                    for sensor, attributes in perception.items()
                    for attribute in attributes
                    if sensor + "_" + attribute not in self.parent_space.members.dtype.names
                ]
            else:
                types = [
                    (sensor + "_" + attribute, float)
                    for sensor, attributes in perception.items()
                    for attribute in attributes
                ]
        return numpy.zeros(size, dtype=types)
    
    def learnable(self):
        """
        Only antipoints are considered learnables.

        :return: Return if the perception (point) is learnable or not.
        :rtype: bool
        """
        for i in self.memberships[0 : self.size]:
            if numpy.isclose(i, -1.0):
                return True
        return False

    @staticmethod
    def copy_perception(space, position, perception):
        """
        Copy a perception to a structured array.

        :param space: An structured array, filled with zeros.
        :type space: numpy.ndarray
        :param position: Position of the array in which the perception is added.
        :type position: int
        :param perception: The perception that is copied in the structured array.
        :type perception: dict
        """
        if getattr(perception, "dtype", None):
            for name in perception.dtype.names:
                if name in space.dtype.names:
                    space[position][name] = perception[name]
        else:
            for sensor, attributes in perception.items():
                for attribute, value in attributes.items():
                    name = sensor + "_" + attribute
                    if name in space.dtype.names:
                        space[position][name] = value

    @staticmethod
    def get_closest_point_and_antipoint_info(members, memberships, foreigner):
        """
        Obtain info about the closest point and antipoint for a given foreigner.

        :param members: Set of the points and antipoints.
        :type members: numpy.ndarray
        :param memberships: The confidence of the points contained in members.
        :type memberships: numpy.ndarray
        :param foreigner: The given foreigner point in order to obtain the info.
        :type foreigner: numpy.ndarray
        :return: The position of in the members array the closest point and antipoints and
            their distance with the foreigner point.
        :rtype: int (position), float (distance)
        """
        distances = numpy.linalg.norm(members - foreigner, axis=1)
        closest_point_pos = None
        closest_point_dist = numpy.finfo(float).max
        closest_antipoint_pos = None
        closest_antipoint_dist = numpy.finfo(float).max
        for pos, _ in enumerate(members):
            if memberships[pos] > 0.0:
                if distances[pos] < closest_point_dist:
                    closest_point_pos = pos
                    closest_point_dist = distances[pos]
            else:
                if distances[pos] < closest_antipoint_dist:
                    closest_antipoint_pos = pos
                    closest_antipoint_dist = distances[pos]
        return closest_point_pos, closest_point_dist, closest_antipoint_pos, closest_antipoint_dist

    def specialize(self, space=None):
        """
        Return a new space with those fields that are in r"space" and not in r"self".

        :param space: Space used to specialize.
        :type space: cognitive_nodes.Space
        :return: The new space.
        :rtype: cognitive_nodes.Space
        """
        new_space = type(self)()
        new_space.parent_space = self
        if space:
            new_space.add_point(space, 1.0)
        return new_space

    def add_point(self, perception, confidence):
        """
        Add a new point to the P-Node.

        :param perception: A given perception to add.
        :type perception: dict
        :param confidence: The confidence of the added point that specifies if it is a point or an
            antipoint.
        :type confidence: float
        :raises RuntimeError: If LTM operation cannot continue.
        :return: The position of the added point.
        :rtype: int
        """
        added_point_pos = -1
        # Currently, we don't add the point if it is an anti-point and the space does not activate for it.
        if (confidence > 0.0) or (self.get_probability(perception) > 0.0):
            if self.parent_space:
                self.parent_space.add_point(perception, confidence)
            # Check if we need to initialize the structured numpy array for storing points
            if self.size == 0:
                # This first point's dtype sets the space's dtype
                # In order to relax this restriction, we will probably replace structured arrays with xarrays
                self.members = self.create_structured_array(perception, None, self.real_size)
                self.memberships = numpy.zeros(self.real_size)
            # Create a new structured array for the new perception
            candidate_point = self.create_structured_array(perception, self.members.dtype, 1)
            # Check if the perception is compatible with this space
            if self.members.dtype != candidate_point.dtype:
                # Node.get_logger().error(
                #     "Trying to add a perception to a NOT compatible space!!!"
                #     "Please, take into account that, at the present time, sensor order in perception matters!!!"
                #     ) #TODO: Pass pnode logger to space
                raise RuntimeError("LTM operation cannot continue :-(")
            else:
                # Copy the new perception on the structured array
                self.copy_perception(candidate_point, 0, perception)
                # Store the new perception if there is a place for it
                if self.size < self.real_size:
                    self.members[self.size] = candidate_point
                    self.memberships[self.size] = confidence
                    added_point_pos = self.size
                    self.size += 1
                else:
                    # Points should be replaced when the P-node is full (may be some metric based on number of times
                    # involved in get_probability)
                    # Node().get_logger().debug(self.ident + " full!")
                    raise RuntimeError("LTM operation cannot continue :-(")
        return added_point_pos

    def get_probability(self, perception):
        """
        Calculate the new activation value.

        :param perception: The given perception to calculate the activation.
        :type perception: dict
        :raises NotImplementedError: The method has to be implemented in a child class.
        """
        raise NotImplementedError

    def contains(self, space, threshold=0.9):
        """
        Check if other space is contained inside this one.
        That happens if this space has a given value of probability for every point belonging to the other space.

        :param space: Space that is checked if it is included.
        :type space: cognitive_nodes.Space
        :param threshold: Minimum probability value.
        :type threshold: float
        :return: Indicates whether the space is contained or not.
        :rtype: bool
        """
        contained = False
        if space.size:
            contained = True
            for point, confidence in zip(space.members[0 : space.size],space.memberships[0 : space.size]) : #Cuando se excluyen los antipuntos????
                self.logger.debug(f"Evaluating point {point} [{confidence}]")
                probability = self.get_probability(point)
                if probability < threshold and confidence>0:
                    self.logger.info(f"Point not contained: {point} ({probability})")
                    contained = False
                    break
        return contained

    def same_sensors(self, space):
        """
        Check if other space has exactly the same sensors that this one.

        :param space: The space to check.
        :type space: cognitive_nodes.Space
        :return: Indicates whether the space has the same sensors or not.
        :rtype: bool
        """
        answer = False
        if self.size and space.size:
            types = [name for name in space.members.dtype.names if name in self.members.dtype.names]
            if len(types) == len(self.members.dtype.names) == len(space.members.dtype.names):
                answer = True
        return answer

    def prune(self, space):
        """
        Prune sensors that are present only in this space or in the space given for comparison.

        :param space: The given space.
        :type space: cognitive_nodes.Space
        """
        common_sensors = [
            (name, float) for name in self.members.dtype.names if name in space.members.dtype.names
        ]
        self.members = require_fields(self.members, common_sensors)

    def aging(self):
        """
        Move towards zero the activation for every point or anti-point.
        """
        for i in range(self.size):
            if self.memberships[i] > 0.0:
                self.memberships[i] -= 0.001
            elif self.memberships[i] < 0.0:
                self.memberships[i] += 0.001
            # This is ugly as it can lead to holes (points that are not really points or antipoints any longer)
            # If this works well, an index structure to reuse these holes should be implemented.
            if numpy.isclose(self.memberships[i], 0.0):
                self.memberships[i] = 0.0


class ClosestPointBasedSpace(PointBasedSpace):
    """
    Calculate the new activation value.

    This activation value is for a given perception and it is calculated as follows:
    - Calculate the closest point to the new point.
    - If the closest point has a positive membership, the membership of the new point is that divided by the distance
    between them. Otherwise, the activation is -1.
    """

    def get_probability(self, perception):
        """
        Calculate the new activation value.

        :param perception: The given perception to calculate the activation.
        :type perception: dict
        :return: The activation value.
        :rtype: float
        """
        # Create a new structured array for the new perception
        candidate_point = self.create_structured_array(perception, self.members.dtype, 1)
        # Copy the new perception on the structured array
        self.copy_perception(candidate_point, 0, perception)
        # Create views on the structured arrays so they can be used in calculations
        members = structured_to_unstructured(
            self.members[0 : self.size][list(candidate_point.dtype.names)]
        )
        point = structured_to_unstructured(candidate_point)
        memberships = self.memberships[0 : self.size]
        # Calculate the activation value
        distances = numpy.linalg.norm(members - point, axis=1)
        pos_closest = numpy.argmin(distances)
        if memberships[pos_closest] > 0.0:
            activation = memberships[pos_closest] / (distances[pos_closest] + 1.0)
        else:
            activation = -1
        return (
            min(activation, self.parent_space.get_probability(perception))
            if self.parent_space
            else activation
        )


class CentroidPointBasedSpace(PointBasedSpace):
    """
    Calculate the new activation value.

    This activation value is for a given perception and it is calculated as follows:
    - Calculate the closest point to the new point.
    - If the closest point has a positive membership, the membership of the new point is that divided by the distance
    between them.
    - Otherwise:
    * Calculate the centroid of points with a positive membership.
    * If the distance from the new point to the centroid is less than the distance from the closest point to the
    centroid, then the activation is calculated as before but using the closest point with positive
    membership. Otherwise the activation is -1.
    """

    def get_probability(self, perception):
        """
        Calculate the new activation value.

        :param perception: The given perception to calculate the activation.
        :type perception: dict
        :return: The activation value.
        :rtype: float
        """
        # Create a new structured array for the new perception
        candidate_point = self.create_structured_array(perception, self.members.dtype, 1)
        # Copy the new perception on the structured array
        self.copy_perception(candidate_point, 0, perception)
        # Create views on the structured arrays so they can be used in calculations
        # Be ware, if candidate_point.dtype is not equal to self.members.dtype, members is a new array!!!
        members = structured_to_unstructured(
            self.members[0 : self.size][list(candidate_point.dtype.names)]
        )
        point = structured_to_unstructured(candidate_point)
        memberships = self.memberships[0 : self.size]
        # Calculate the activation value
        distances = numpy.linalg.norm(members - point, axis=1)
        pos_closest = numpy.argmin(distances)
        if memberships[pos_closest] > 0.0:
            activation = memberships[pos_closest] / (distances[pos_closest] + 1.0)
        else:
            centroid = numpy.mean(members[memberships > 0.0], axis=0)
            dist_antipoint_centroid = numpy.linalg.norm(members[pos_closest] - centroid)
            dist_newpoint_centroid = numpy.linalg.norm(point - centroid)
            if dist_newpoint_centroid + 0.000001 < dist_antipoint_centroid:
                distances = distances[memberships > 0.0]
                pos_closest = numpy.argmin(distances)
                memberships = memberships[memberships > 0.0]
                activation = memberships[pos_closest] / (distances[pos_closest] + 1.0)
            else:
                activation = -1
        return (
            min(activation, self.parent_space.get_probability(perception))
            if self.parent_space
            else activation
        )


class NormalCentroidPointBasedSpace(PointBasedSpace):
    """
    Calculate the new activation value.

    This activation value is for a given perception and it is calculated as follows:
    - Calculate the closest point to the new point.
    - If the closest point has a positive membership, the membership of the new point is that divided by the distance
    between them.
    - Otherwise:
    * Calculate the centroid of points with a positive membership.
    * If the distance from the new point to the centroid is less than the distance from the closest point to the
    centroid, or the distance of the closest point to the line that goes from the new point to the centroid is high
    (see source code), then the activation is calculated as before but using the closest point with positive
    membership, otherwise the activation is -1.
    """

    def get_probability(self, perception):
        """
        Calculate the new activation value.

        :param perception: The given perception to calculate the activation.
        :type perception: dict
        :return: The activation value.
        :rtype: float
        """
        # Create a new structured array for the new perception
        candidate_point = self.create_structured_array(perception, self.members.dtype, 1)
        # Copy the new perception on the structured array
        self.copy_perception(candidate_point, 0, perception)
        # Create views on the structured arrays so they can be used in calculations
        # Be ware, if candidate_point.dtype is not equal to self.members.dtype, members is a new array!!!
        members = structured_to_unstructured(
            self.members[0 : self.size][list(candidate_point.dtype.names)]
        )
        point = structured_to_unstructured(candidate_point)
        memberships = self.memberships[0 : self.size]
        # Calculate the activation value
        distances = numpy.linalg.norm(members - point, axis=1)
        pos_closest = numpy.argmin(distances)
        if memberships[pos_closest] > 0.0:
            activation = memberships[pos_closest] / (distances[pos_closest] + 1.0)
        else:
            centroid = numpy.mean(members[memberships > 0.0], axis=0)
            v_antipoint_centroid = numpy.ravel(members[pos_closest] - centroid)
            v_newpoint_centroid = numpy.ravel(point - centroid)
            dist_antipoint_centroid = numpy.linalg.norm(v_antipoint_centroid)
            dist_newpoint_centroid = numpy.linalg.norm(v_newpoint_centroid)
            # https://en.wikipedia.org/wiki/Vector_projection
            separation = numpy.linalg.norm(
                v_antipoint_centroid
                - numpy.inner(
                    v_newpoint_centroid,
                    numpy.inner(v_antipoint_centroid, v_newpoint_centroid)
                    / numpy.inner(v_newpoint_centroid, v_newpoint_centroid),
                )
            )
            if (dist_newpoint_centroid < dist_antipoint_centroid) or (
                numpy.random.uniform()
                < dist_antipoint_centroid * separation / dist_newpoint_centroid
            ):
                distances = distances[memberships > 0.0]
                pos_closest = numpy.argmin(distances)
                memberships = memberships[memberships > 0.0]
                activation = memberships[pos_closest] / (distances[pos_closest] + 1.0)
            else:
                activation = -1
        return (
            min(activation, self.parent_space.get_probability(perception))
            if self.parent_space
            else activation
        )
    
class ActivatedDummySpace(PointBasedSpace):
    """
    A dummy space that always returns an activation of 1.0 for any perception.
    """
    def add_point(self, perception, confidence):
        """
        Dummy method to add a point to the space.
        This method does not actually add any points.

        :param perception: A given perception to add. It is not used.
        :type perception: dict
        :param confidence: The confidence of the added point. Irrelevant in this case.
        :type confidence: float
        :return: -1
        :rtype: int
        """
        return -1

    def get_probability(self, perception):
        """
        Activation value is always 1.0.

        :param perception: A given perception to add. It is not used.
        :type perception: dict
        :return: The activation value, which is always 1.0.
        :rtype: float
        """
        return 1.0

class SVMSpace(PointBasedSpace):
    """
    Use a SVM to calculate activations.
    """

    def __init__(self, **kwargs):
        """
        Init attributes when a new object is created.
        """
        self.model = svm.SVC(kernel="poly", degree=32, max_iter=200000)
        super().__init__(**kwargs)


    def prune_points(self, score, memberships):
        """
        Prune points depending on the model score obtained.

        :param score: Score that determines the pruning.
        :type score: float
        :param memberships: The confidence of the points.
        :type memberships: numpy.ndarray
        """
        if numpy.isclose(score, 1.0):
            self.size = len(self.model.support_vectors_)
            for i, vector in zip(self.model.support_, self.model.support_vectors_):
                self.members[i] = tuple(vector)
                self.memberships[i] = memberships[i]

    def fit_and_score(self):
        """
        Fit and score the SVM Model.

        :return: The score of the model.
        :rtype: float
        """
        members = structured_to_unstructured(
            self.members[0 : self.size][list(self.members.dtype.names)]
        )
        memberships = self.memberships[0 : self.size].copy()
        memberships[memberships > 0] = 1
        memberships[memberships <= 0] = 0
        self.model.fit(members, memberships)
        score = self.model.score(members, memberships)
        self.logger.debug(
            "SVM: iterations "
            + str(self.model.n_iter_)
            + " support vectors "
            + str(len(self.model.support_vectors_))
            + " score "
            + str(score)
            + " points "
            + str(len(members))
        ) #TODO: Pass pnode logger to space
        return score

    def remove_close_points(self):
        """
        Remove points that are too close in space.
        """
        threshold = 0
        previous_size = self.size
        members = self.members[0 : self.size].copy()
        umembers = structured_to_unstructured(members[list(self.members.dtype.names)])
        memberships = self.memberships[0 : self.size].copy()
        score = 0.3
        while score < 1.0:
            threshold += 0.1
            distances = numpy.linalg.norm(umembers - umembers[previous_size - 1], axis=1)
            indexes = distances > threshold
            filtered_members = members[indexes]
            filtered_memberships = memberships[indexes]
            self.size = len(filtered_members)
            if self.size < previous_size - 1:
                for i in range(self.size):
                    self.members[i] = filtered_members[i]
                    self.memberships[i] = filtered_memberships[i]
                self.members[self.size] = members[previous_size - 1]
                self.memberships[self.size] = memberships[previous_size - 1]
                self.size += 1
                score = self.fit_and_score()
        # Node.get_logger.logdebug(
        #     self.ident + ": throwing away " + str(previous_size - self.size) + " points."
        # ) #TODO: Pass pnode logger to space

    def add_point(self, perception, confidence):
        """
        Add a new point to the P-Node.

        :param perception: A given perception to add.
        :type perception: dict
        :param confidence: The confidence of the added point that specifies if it is a point or an
            antipoint.
        :type confidence: float
        :return: The position of the added point.
        :rtype: int
        """
        pos = super().add_point(perception, confidence)
        if self.learnable():
            self.fit_and_score()
        prediction = self.get_probability(perception)
        if ((confidence > 0.0) and (prediction <= 0.0)) or (
            (confidence <= 0.0) and (prediction > 0.0)
        ):
            if self.fit_and_score() < 1.0:
                self.remove_close_points()
        return pos

    def get_probability(self, perception):
        """
        Calculate the new activation value.

        :param perception: The given perception to calculate the activation.
        :type perception: dict
        :return: The activation value.
        :rtype: float
        """
        # Create a new structured array for the new perception
        candidate_point = self.create_structured_array(perception, self.members.dtype, 1)
        # Copy the new perception on the structured array
        self.copy_perception(candidate_point, 0, perception)
        # Create views on the structured arrays so they can be used in calculations
        # Beware, if candidate_point.dtype is not equal to self.members.dtype, members is a new array!
        point = structured_to_unstructured(candidate_point)
        # Calculate the activation value
        if self.learnable():
            act = min(2.0, self.model.decision_function(point)[0]) / 2.0
        else:
            act = 1.0
        return min(act, self.parent_space.get_probability(perception)) if self.parent_space else act


class ANNSpace(PointBasedSpace):
    """
    Use and train a Neural Network to calculate the activations.
    """

    def __init__(self, **kwargs):
        """
        Init attributes when a new object is created.
        """
        #GPU USAGE TEST
        tf.config.set_visible_devices([], 'GPU') #Temporary disable of GPU
        '''
        #tf.debugging.set_log_device_placement(True) #Detailed log in every TF operation
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Set memory growth to avoid allocating all GPU memory
                for gpu in gpus:
                    tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
                )
            except RuntimeError as e:
                print(e)
        '''

        # self.n_splits = 5
        self.batch_size = 50
        self.epochs = 50
        self.max_data = 400
        self.first_data = 0
        # Define the Neural Network's model
        self.model = None

        # Initialize variables
        self.there_are_points = False
        self.there_are_antipoints = False
        super().__init__(**kwargs)

    def build_model(self, input_shape):
        """
        Build the model with the given input shape.

        :param input_shape: The shape of the input data.
        :type input_shape: tuple
        """
        # Define train values
        output_activation = "sigmoid"
        optimizer = tf.optimizers.Adam()
        loss = tf.losses.BinaryCrossentropy()
        metrics = ["accuracy"]

        # Build the model
        model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation=output_activation),
        ])

        # Compile the model
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # Log the model summary for debugging purposes
        self.logger.debug(f"Model summary for {self.ident}:")
        model.summary(print_fn=self.logger.debug)

        # Return the compiled model  
        return model

    def add_point(self, perception, confidence):
        """
        Add a new point to the P-Node.

        :param perception: A given perception to add.
        :type perception: dict
        :param confidence: The confidence of the added point that specifies if it is a point or an
            antipoint.
        :type confidence: float
        :return: The position of the added point.
        :rtype: int
        """
        pos = None

        if confidence > 0.0:
            self.there_are_points = True
        else:
            self.there_are_antipoints = True

        if self.there_are_points and self.there_are_antipoints:
            candidate_point = self.create_structured_array(perception, self.members.dtype, 1)
            self.copy_perception(candidate_point, 0, perception)
            point = tf.convert_to_tensor(structured_to_unstructured(candidate_point))
            
            # If the model is not built yet, build it
            if self.model is None:
                input_shape = point.shape[1]  # Get the number of features from the point
                self.model = self.build_model(input_shape)

            # Catch diff between point and model input shape
            if point.shape[1] != self.model.input_shape[1]:
                self.logger.error(
                    f"Point shape {point.shape} does not match model input shape {self.model.input_shape}"
                )
                raise RuntimeError("LTM operation cannot continue :-(")    

            prediction = (self.model.call(point)[0][0]*2)-1 #Pass from [0,1] to [-1, 1]
            pos = super().add_point(perception, confidence)

            members = structured_to_unstructured(
                self.members[0 : self.size][list(self.members.dtype.names)]
            )
            memberships = self.memberships[0 : self.size].copy()
            memberships[memberships > 0] = 1.0
            memberships[memberships <= 0] = 0.0


            if self.size >= self.max_data:
                self.first_data = self.size - self.max_data

            if True: #HACK: Select a proper training threshold
                # Node.get_logger().logdebug(f"Training... {self.ident}") #TODO: Pass pnode logger to space
                self.logger.info(f"[TRAINING] {self.ident}: point added with confidence {confidence}, prediction {prediction}")
                X = members[self.first_data : self.size]
                Y = memberships[self.first_data : self.size]
                n_0 = int(len(Y[Y == 0.0]))
                n_1 = int(len(Y[Y == 1.0]))
                weight_for_0 = (
                    (1 / n_0) * ((self.size - self.first_data) / 2.0) if n_0 != 0 else 1.0
                )
                weight_for_1 = (
                    (1 / n_1) * ((self.size - self.first_data) / 2.0) if n_1 != 0 else 1.0
                )
                class_weight = {0: weight_for_0, 1: weight_for_1}
                self.model.fit(
                    x=X,
                    y=Y,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    verbose=0,
                    class_weight=class_weight,
                )

        else:
            pos = super().add_point(perception, confidence)

        return pos

    def get_probability(self, perception):
        """
        Calculate the new activation value.

        :param perception: The given perception to calculate the activation.
        :type perception: dict
        :return: The activation value.
        :rtype: float
        """
        candidate_point = self.create_structured_array(perception, self.members.dtype, 1)
        self.copy_perception(candidate_point, 0, perception)
        point = tf.convert_to_tensor(structured_to_unstructured(candidate_point))
        if self.there_are_points:
            if self.there_are_antipoints:
                act = float(self.model.call(point)[0][0])
                if act < 0.01:
                    act=0.0
            else:
                act = 1.0
        else:
            act = 0.0
        return min(act, self.parent_space.get_probability(perception)) if self.parent_space else act


class PyTorchOnlineSGDSpace(PointBasedSpace):
    """
    SGD Online: P(confidence=1 | percepción) vs confidence=-1.
    Entrena con cada nuevo punto/antipunto en streaming.
    """
    
    def __init__(self, lr=0.05, momentum=0.9, window_size=200, **kwargs):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        self.torch = torch
        self.nn = nn
        self.optim = optim
        
        self.lr = lr
        self.momentum = momentum
        self.window_size = window_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.optimizer = None
        self.feature_keys = None
        
        # NUEVO: Guarda percepciones originales + features + labels
        self.perception_history = []  # [(perception_dict, features, label), ...]
        
        super().__init__(**kwargs)
        self.logger.info(f"SGD Online (1=punto, -1=antipunto) lr={lr} on {self.device}")
    
    def _flatten_perception(self, perception):
        """Dict anidado → vector features ordenado."""
        import numpy as np
        self.logger.debug(f"[SGD] flatten start type={type(perception)}")
        if isinstance(perception, np.void):
            perception = {name: perception[name] for name in perception.dtype.names}
        
        features = []
        keys = []
        
        for sensor_name in sorted(perception.keys()):
            sensor_data = perception[sensor_name]
            if isinstance(sensor_data, dict):
                for field in sorted(sensor_data.keys()):
                    val = sensor_data[field]
                    features.append(float(val) if val is not None else 0.0)
                    keys.append(f"{sensor_name}.{field}")
            elif isinstance(sensor_data, list) and len(sensor_data) > 0 and isinstance(sensor_data[0], dict):
                # Maneja listas de dicts como {'data': value}
                for field in sorted(sensor_data[0].keys()):
                    val = sensor_data[0][field]
                    features.append(float(val) if val is not None else 0.0)
                    keys.append(f"{sensor_name}.{field}")
            else:
                features.append(float(sensor_data))
                keys.append(sensor_name)
        
        if self.feature_keys is None:
            self.feature_keys = keys
            self.logger.info(f"[SGD] Features init ({len(keys)}): {keys[:8]}...")
        self.logger.debug(f"[SGD] flatten done feats={len(features)}")
        return np.array(features, dtype=np.float32)
    
    def build_model(self, input_dim):
        self.logger.info(f"[SGD] build_model input_dim={input_dim}")
        nn_ref = self.nn
        torch_ref = self.torch
        
        class ConfidenceNet(nn_ref.Module):
            def __init__(self, input_size):
                super().__init__()
                self.ln = nn_ref.LayerNorm(input_size)
                self.fc1 = nn_ref.Linear(input_size, 64)
                self.fc2 = nn_ref.Linear(64, 32)
                self.out = nn_ref.Linear(32, 1)
            
            def forward(self, x):
                x = self.ln(x)
                x = torch_ref.relu(self.fc1(x))
                x = torch_ref.relu(self.fc2(x))
                return self.out(x)
        
        return ConfidenceNet(input_dim).to(self.device)
    
    def add_point(self, perception, confidence):
        """Añade punto + guarda en histórico."""
        import numpy as np
        self.logger.debug(f"[SGD] add_point conf={confidence}")
        pos = super().add_point(perception, confidence)
        if pos < 0:
            self.logger.debug("[SGD] super.add_point rejected")
            return pos
        
        # NUEVO: Extrae features y guarda en histórico
        try:
            feats = self._flatten_perception(perception)
            label = 1.0 if confidence > 0 else 0.0
            self.perception_history.append((perception, feats, label))
            self.logger.debug(f"[SGD] hist size={len(self.perception_history)} feats_len={len(feats)}")
        except Exception as e:
            self.logger.warning(f"[SGD] Error flattening perception: {e}")
            return pos

        if self.model is None and len(self.perception_history) > 2:
            self.model = self.build_model(feats.shape[0])
            self.optimizer = self.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
            self.logger.info("[SGD] model created")

        if self.model is not None and len(self.perception_history) > 10:
            X, Y = self._get_training_data()
            pos_count = np.sum(Y == 1.0)
            neg_count = np.sum(Y == 0.0)
            self.logger.debug(f"[SGD] training window X={X.shape} pos={pos_count} neg={neg_count}")
            if pos_count > 0 and neg_count > 0:
                self._sgd_online_update(X, Y)
                self.logger.info(f"[SGD] SGD done pos={pos_count} neg={neg_count}")
        return pos

    def _get_training_data(self):
        import numpy as np
        start = max(0, len(self.perception_history) - self.window_size)
        window = self.perception_history[start:]
        X = np.array([feats for _, feats, _ in window])
        Y = np.array([label for _, _, label in window])
        self.logger.debug(f"[SGD] get_training_data X={X.shape} Y={Y.shape}")
        return X, Y

    def _sgd_online_update(self, X, Y):
        self.logger.debug(f"[SGD] _sgd_online_update start batches={max(1, len(X)//8)}")
        X_t = self.torch.tensor(X, dtype=self.torch.float32).to(self.device)
        Y_t = self.torch.tensor(Y, dtype=self.torch.float32).unsqueeze(1).to(self.device)
        criterion = self.nn.BCEWithLogitsLoss()
        self.model.train()
        indices = self.torch.randperm(len(X_t))
        total_loss = 0
        num_batches = 0
        for i in range(0, len(X_t), 8):
            batch_idx = indices[i:min(i+8, len(X_t))]
            batch_X, batch_Y = X_t[batch_idx], Y_t[batch_idx]
            logits = self.model(batch_X)
            loss = criterion(logits, batch_Y)
            self.optimizer.zero_grad()
            loss.backward()
            self.torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.logger.debug(f"[SGD] _sgd_online_update done avg_loss={avg_loss:.4f}")

    def get_probability(self, perception):
        """P(confidence=1) para nueva percepción."""
        if self.model is None:
            self.logger.debug("[SGD] get_prob model=None =>0.5")
            return 0.5
        try:
            feats = self._flatten_perception(perception)
        except Exception as e:
            self.logger.error(f"[SGD] Error in get_probability: {e}")
            return 0.5
        
        self.model.eval()
        with self.torch.no_grad():
            x = self.torch.tensor(feats, dtype=self.torch.float32).unsqueeze(0).to(self.device)
            logit = self.model(x)
            prob_success = self.torch.sigmoid(logit).item()
        
        self.logger.debug(f"[SGD] get_prob={prob_success:.4f}")
        return prob_success
