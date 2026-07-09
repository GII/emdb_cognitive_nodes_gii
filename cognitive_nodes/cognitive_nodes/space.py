from __future__ import annotations

from math import isclose
import os
import numpy as np
import threading
from numpy.lib.recfunctions import structured_to_unstructured, require_fields
import pandas as pd
from sklearn import svm
from rclpy.node import Node
from rclpy.logging import get_logger

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from core.utils import separate_perceptions, resolve_seed
from core.container import Container
from cognitive_nodes.random_utils import set_global_seeds

from core_interfaces.msg import Container as ContainerMsg

class Space(object):
    """A n-dimensional state space."""

    def __init__(self, ident=None, random_seed=0, **kwargs):
        """Init attributes when a new object is created.

        :param ident: The name of the space.
        :type ident: str
        """
        self.ident = ident
        self.parent_space = None
        # random_seed 0/None -> fresh time-based seed (genuinely random run).
        self.random_seed = resolve_seed(random_seed)
        self.logger = get_logger("space_" + str(ident))
        self.logger.info(f"CREATING SPACE: {ident}")
        self.rng = np.random.default_rng(self.random_seed)


class PointBasedSpace(Space):
    """A state space based on points."""

    def __init__(self, size=30000, **kwargs):
        """
        Init attributes when a new object is created.

        :param size: Maximum number of points that the space can contain, defaults to 5000.
        :type size: int
        """
        self._data = None
        self.real_size = size
        super().__init__(**kwargs)

    @property
    def size(self):
        """
        Get the size of the space.

        :return: The size of the space.
        :rtype: int
        """
        return self._data.size if self._data is not None else 0

    @property
    def members(self):
        """
        Get the members of the space.

        :return: The members of the space.
        :rtype: list
        """
        if self._data is not None:        
            return self._data.read(ordered=True).drop_sel(features=["confidence"]).values
        else:
            return np.array([])
    
    @property
    def memberships(self):
        """
        Get the memberships of the space.

        :return: The memberships of the space.
        :rtype: list
        """
        if self._data is not None:
            return self._data.read(ordered=True).sel(features=["confidence"]).values
        else:
            return np.array([])

    @classmethod
    def populate_space(cls, data_container: Container, **kwargs) -> PointBasedSpace:
        """
        Populates the space from the data container. 

        NOTE: This method is intended to be used when loading a space from a saved state, where the data container already contains the necessary information to populate the space. (e.g. it has the confidences stored as a column, so we don't need to pass them separately as in add_point)

        :param point: A perception dictionary describing the structure of the space.
        :type point: dict
        :param members: A flattened list of data with size n_dims * n_data.
        :type members: list
        :param memberships: A list of membership data with size n_data.
        :type memberships: list
        :raises ValueError: If the size of memberships does not match the calculated size of the space.
        """
        data_name = data_container.name
        if data_name.endswith("_data"):
            ident = data_name[:-5]  # Remove the "_data" suffix to get the space identifier
        else:
            ident = data_name
            data_container.name = ident + "_data"  # Ensure the container name follows the expected format
        size = len(data_container)
        space = cls(ident=ident, size=size, **kwargs)
        space._data = data_container
        return space

    def initialize_data_structure(self, point: Container, size): 
        """
        Initialize the structured array based on the given point and size.
        :param point: A perception that will provide the structure for the space.
        :type point: Container
        :param size: The size of the structured array.
        :type size: int
        """
        labels = point.feature_labels + ["confidence"]
        self._data = Container(name=self.ident + "_data", max_size=size, container_type="space", labels=labels)

    def learnable(self):
        """
        Only antipoints are considered learnables.

        :return: Return if the perception (point) is learnable or not.
        :rtype: bool
        """
        for i in self.memberships[0 : self.size]:
            if np.isclose(i, -1.0):
                return True
        return False
    
    def reload_members(self, members, memberships, timestamps):
        """ Reload the members and memberships of the space with new values."""
        self._data.clear()
        data = np.concatenate([members, memberships.reshape(-1, 1)], axis=1)
        labels = self._data.feature_labels
        self._data.push(data, labels, timestamps=timestamps)

    def data_from_perception(self, perception: Container):
        """
        Create a structured array for the given perception.

        :param perception: The given perception to create the structured array.
        :type perception: Container
        :return: The structured array created from the given perception.
        :rtype: numpy.ndarray
        """
        data = perception.read(ordered=True)
        filtered_features = data.sel(features=self._data.feature_labels[:-1])
        return filtered_features.values

    def to_msg(self):
        """
        Convert the space to a message format.
        
        :return: The message format of the space.
        :rtype: ContainerMsg
        """
        if self._data is not None:
            return self._data.to_msg()
        else:
            self.logger.warning(f"Trying to convert space {self.ident} to message, but it has no data.")
            msg = ContainerMsg()
            msg.name = self.ident + "_data" 
            return msg

    @staticmethod
    def get_closest_point_and_antipoint_info(members, memberships, foreigner):
        """
        Obtain info about the closest point and antipoint for a given foreigner.

        :param members: Set of the points and antipoints.
        :type members: np.ndarray
        :param memberships: The confidence of the points contained in members.
        :type memberships: numpy.ndarray
        :param foreigner: The given foreigner point in order to obtain the info.
        :type foreigner: numpy.ndarray
        :return: The position of in the members array the closest point and antipoints and
            their distance with the foreigner point.
        :rtype: int (position), float (distance)
        """
        distances = np.linalg.norm(members - foreigner, axis=1)
        closest_point_pos = None
        closest_point_dist = np.finfo(float).max
        closest_antipoint_pos = None
        closest_antipoint_dist = np.finfo(float).max
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

    def add_point(self, perceptions: Container, confidences: np.ndarray):
        """
        Add a new point to the P-Node.

        :param perception: A given perception to add.
        :type perception: Container
        :param confidence: The confidence of the added point that specifies if it is a point or an
            antipoint.
        :type confidence: float
        :raises RuntimeError: If LTM operation cannot continue.
        :return: The position of the added point.
        :rtype: int | numpy.ndarray
        """
        added_point_pos = -1
        # Currently, we don't add the point if it is an anti-point and the space does not activate for it.
        probabilities = self.get_probability(perceptions)
        ## Create an index list for the points that either have a confidence greater than 0.0 or a probability greater than 0.0
        indexes = (confidences > 0.0) | (probabilities > 0.0)
        data_array = perceptions.read(ordered=True)
        data = data_array.values[indexes]
        timestamps = data_array.coords["timestamp"].values[indexes]
        # Add the confidence as a new column to the data array, so it can be stored in the structured array of the space
        points = np.concatenate([data, confidences[indexes].reshape(-1, 1)], axis=1)
        labels = perceptions.feature_labels + ["confidence"]

        if points.shape[0] > 0:
            if self.parent_space:
                self.parent_space.add_point(perceptions, confidences)
            # Check if we need to initialize the container for storing points
            if self.size == 0:
                self.initialize_data_structure(perceptions, self.real_size)
            # Add the new points to the container
            added_point_pos = self._data.push(points, labels, timestamps=timestamps)
        return added_point_pos

    def get_probability(self, perceptions):
        """
        Calculate the new activation value.

        :param perceptions: The given perceptions to calculate the activation.
        :type perceptions: core.container.Container
        :raises NotImplementedError: The method has to be implemented in a child class.
        """
        raise NotImplementedError

    def contains(self, space_data: Container, threshold=0.9):
        """
        Check if other space is contained inside this one.
        That happens if this space has a given value of probability for every point belonging to the other space.
        """
        data_name = space_data.name[:-5] if space_data.name.endswith("_data") else space_data.name
        self.logger.info(f"Checking if space {space_data.name} is contained with threshold {threshold}")

        if space_data.size <= 0:
            self.logger.info(f"Space {data_name} is empty.")
            return False

        data_array = space_data.read(ordered=True)
        point_confidences = data_array.sel(features=["confidence"]).values.reshape(-1)
        positive_idx = np.flatnonzero(point_confidences > 0.0)

        if positive_idx.size == 0:
            self.logger.info(f"Space {data_name} contains only antipoints; skipping containment checks.")
            return True

        positive_data = Container.from_dataarray(
            data_array.isel(sample=positive_idx),
            container_type=space_data.container_type,
            name=space_data.name,
        )

        probabilities = self.get_probability(positive_data)

        if np.all(probabilities >= threshold):
            self.logger.info(
                f"Space {data_name} is contained with mean probability {np.mean(probabilities)}"
            )
            return True

        self.logger.info(f"Space {data_name} has points that are not contained.")
        for pos, prob in zip(positive_idx, probabilities):
            if prob < threshold:
                self.logger.info(
                    f"Point at position {pos} with confidence {point_confidences[pos]} "
                    f"and probability {prob} is not contained."
                )
        return False

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
            self_labels = set(self._data.feature_labels)
            compared_labels = set(space._data.feature_labels)
            if self_labels == compared_labels:
                answer = True
        return answer

    def prune(self, space):
        """
        Prune sensors that are present only in this space or in the space given for comparison.

        :param space: The given space.
        :type space: cognitive_nodes.Space
        """
        common_sensors = [
            name for name in self._data.feature_labels if name in space._data.feature_labels
        ]
        data_array = self._data.read(ordered=True)
        data = data_array.sel(features=common_sensors).values
        timestamps = data_array.coords["timestamps"].values
        self._data = Container(name=self.ident + "_data", max_size=self.real_size, container_type="space", labels=common_sensors)
        self._data.push(data, common_sensors, timestamps=timestamps)

class ClosestPointBasedSpace(PointBasedSpace):
    """
    Calculate the new activation value.

    This activation value is for a given perception and it is calculated as follows:
    - Calculate the closest point to the new point.
    - If the closest point has a positive membership, the membership of the new point is that divided by the distance
    between them. Otherwise, the activation is -1.
    """
    def get_probability(self, perceptions):
        """
        Calculate the new activation value for multiple perception rows.

        :param perceptions: The given perceptions to calculate the activation.
        :type perceptions: core.container.Container
        :return: The activation values, one per perception row.
        :rtype: np.ndarray
        """
        # Obtain the datapoint from the given perception (selects the appropriate features)
        if self._data is not None:
            points = self.data_from_perception(perceptions)
        else:
            return np.zeros(perceptions.size, dtype=float)
        # Obtain the members and memberships of the space
        members = self.members
        memberships = self.memberships
        # Calculate the activation value
        n_rows = points.shape[0]
        # No stored points yet -> no activation
        if members.size == 0 or memberships.size == 0:
            activation = np.zeros(n_rows, dtype=float)
        else:
            # members: shape (None, n_members, n_features)
            # points: shape (n_rows, None, n_features)
            # distances: shape (n_rows, n_members)
            distances = np.linalg.norm(members[None, :, :] - points[:, None, :], axis=2)

            pos_closest = np.argmin(distances, axis=1)  # one closest member per row
            closest_dist = distances[np.arange(n_rows), pos_closest]
            closest_membership = memberships[pos_closest]

            activation = np.where(
                closest_membership > 0.0,
                closest_membership / (closest_dist + 1.0),
                -1.0,
            )
        if self.parent_space:
            parent_act = self.parent_space.get_probability(perceptions)
            activation = np.minimum(activation, parent_act)
        return activation.reshape(-1)


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

    def get_probability(self, perceptions):
        """
        Calculate the new activation value for multiple perception rows.

        :param perceptions: The given perceptions to calculate the activation.
        :type perceptions: core.container.Container
        :return: The activation values, one per perception row.
        :rtype: np.ndarray
        """
        # Obtain the datapoint from the given perception (selects the appropriate features)
        if self._data is not None:
            points = self.data_from_perception(perceptions)
        else:
            return np.zeros(perceptions.size, dtype=float)
        # Obtain the members and memberships of the space
        members = self.members
        memberships = self.memberships
        # Calculate the activation value
        n_rows = points.shape[0]
        # No stored points yet -> no activation
        if members.size == 0 or memberships.size == 0:
            activation = np.zeros(n_rows, dtype=float)
        else:
            # Compute pairwise distances: shape (n_rows, n_members)
            distances = np.linalg.norm(members[None, :, :] - points[:, None, :], axis=2)
            
            pos_closest = np.argmin(distances, axis=1)
            closest_dist = distances[np.arange(n_rows), pos_closest]
            closest_membership = memberships[pos_closest]
            
            # Vectorized case 1: closest member is positive
            activation = np.where(
                closest_membership > 0.0,
                closest_membership / (closest_dist + 1.0),
                -1.0,
            )
            
            # Handle case 2: closest member is negative (antipoint)
            neg_mask = closest_membership <= 0.0
            
            if np.any(neg_mask) and np.any(memberships > 0.0):
                # Centroid of points with positive membership
                positive_indexes = (memberships > 0.0).reshape(-1)
                centroid = np.mean(members[positive_indexes], axis=0)
                
                # Distance from each point to centroid
                dist_points_centroid = np.linalg.norm(points - centroid[None, :], axis=1)
                
                # Distance from each closest antipoint to centroid
                dist_antipoints_centroid = np.linalg.norm(members[pos_closest] - centroid[None, :], axis=1)
                
                # Points closer to centroid than antipoint
                closer_mask = (dist_points_centroid + 0.000001) < dist_antipoints_centroid
                update_mask = neg_mask & closer_mask
                
                if np.any(update_mask):
                    pos_members = members[positive_indexes]
                    pos_memberships = memberships[positive_indexes]
                    
                    for row_idx in np.where(update_mask)[0]:
                        dists = np.linalg.norm(pos_members - points[row_idx], axis=1)
                        closest_pos_idx = np.argmin(dists)
                        activation[row_idx] = pos_memberships[closest_pos_idx] / (dists[closest_pos_idx] + 1.0)
        
        if self.parent_space:
            parent_act = self.parent_space.get_probability(perceptions)
            activation = np.minimum(activation, parent_act)
        
        return activation.reshape(-1)


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

    def get_probability(self, perceptions):
        """
        Calculate the new activation value for multiple perception rows.

        :param perceptions: The given perceptions to calculate the activation.
        :type perceptions: core.container.Container
        :return: The activation values, one per perception row.
        :rtype: np.ndarray
        """
        # Obtain the datapoint from the given perception (selects the appropriate features)
        if self._data is not None:
            points = self.data_from_perception(perceptions)
        else:
            return np.zeros(perceptions.size, dtype=float)
        # Obtain the members and memberships of the space
        members = self.members
        memberships = self.memberships
        # Calculate the activation value
        n_rows = points.shape[0]
        # No stored points yet -> no activation
        if members.size == 0 or memberships.size == 0:
            activation = np.zeros(n_rows, dtype=float)
        else:
            # Compute pairwise distances: shape (n_rows, n_members)
            distances = np.linalg.norm(members[None, :, :] - points[:, None, :], axis=2)
            
            pos_closest = np.argmin(distances, axis=1)
            closest_dist = distances[np.arange(n_rows), pos_closest]
            closest_membership = memberships[pos_closest]
            
            # Vectorized case 1: closest member is positive
            activation = np.where(
                closest_membership > 0.0,
                closest_membership / (closest_dist + 1.0),
                -1.0,
            )
            
            # Handle case 2: closest member is negative (antipoint)
            neg_mask = closest_membership <= 0.0
            if np.any(neg_mask) and np.any(memberships > 0.0):
                # Centroid of points with positive membership
                positive_indexes = (memberships > 0.0).reshape(-1)
                centroid = np.mean(members[positive_indexes], axis=0)

                # Distance from each point to centroid
                dist_newpoint_centroid = np.linalg.norm(points - centroid[None, :], axis=1)
                # Distance from each closest antipoint to centroid
                v_antipoint_centroid = members[pos_closest] - centroid[None, :]
                dist_antipoint_centroid = np.linalg.norm(v_antipoint_centroid, axis=1)

                # Vector from new points to centroid
                v_newpoint_centroid = points - centroid[None, :]

                # Check if new point is closer to centroid than antipoint, or if antipoint is far from the line between new point and centroid
                # https://en.wikipedia.org/wiki/Vector_projection
                dot_num = np.sum(v_antipoint_centroid * v_newpoint_centroid, axis=1)
                dot_den = np.sum(v_newpoint_centroid * v_newpoint_centroid, axis=1)
                projection = v_newpoint_centroid * (dot_num / dot_den)[:, None]
                separation = np.linalg.norm(v_antipoint_centroid - projection, axis=1)

                update_mask = neg_mask & (
                    (dist_newpoint_centroid < dist_antipoint_centroid)
                    | (
                        self.rng.uniform(size=n_rows)
                        < (dist_antipoint_centroid * separation / dist_newpoint_centroid)
                    )
                )

                if np.any(update_mask):
                    positive_members = members[positive_indexes]
                    positive_memberships = memberships[positive_indexes]

                    for row_idx in np.where(update_mask)[0]:
                        row_distances = np.linalg.norm(positive_members - points[row_idx], axis=1)
                        pos_closest_positive = np.argmin(row_distances)
                        activation[row_idx] = (
                            positive_memberships[pos_closest_positive]
                            / (row_distances[pos_closest_positive] + 1.0)
                        )
        if self.parent_space:
            parent_act = self.parent_space.get_probability(perceptions)
            activation = np.minimum(activation, parent_act)
        
        return activation.reshape(-1)
    
class ActivatedDummySpace(PointBasedSpace):
    """
    A dummy space that always returns an activation of 1.0 for any perception.
    """
    def add_point(self, perceptions, confidences):
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

    def get_probability(self, perceptions):
        """
        Calculate the new activation value for multiple perception rows.

        :param perceptions: The given perceptions to calculate the activation.
        :type perceptions: core.container.Container
        :return: The activation values, one per perception row.
        :rtype: np.ndarray
        """
        return np.ones(perceptions.size, dtype=float)

class SVMSpace(PointBasedSpace):
    """
    Use a SVM to calculate activations.
    """

    def __init__(self, kernel="poly", degree=32, max_iter=200000, **kwargs):
        """
        Init attributes when a new object is created.
        """
        # random_seed is read from kwargs because super().__init__ (which stores
        # self.random_seed) has not run yet at this point.
        self.model = svm.SVC(kernel=kernel, degree=degree, max_iter=max_iter, random_state=resolve_seed(kwargs.get('random_seed')))
        super().__init__(**kwargs)

    def fit_and_score(self):
        """
        Fit and score the SVM Model.

        :return: The score of the model.
        :rtype: float
        """
        members = self.members
        memberships = self.memberships.copy().reshape(-1)
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
        )
        return score

    def remove_close_points(self):
        """
        Remove points that are too close in space.
        """
        threshold = 0
        previous_size = self.size
        members = self.members.copy()
        memberships = self.memberships.copy()
        timestamps = self._data.read(ordered=True).coords["timestamp"].values.copy()
        score = 0.3
        while score < 1.0:
            # Adjusted threshold to be more aggressive in removing close points in each iteration
            threshold += 0.1
            # Calculate distances from the last member to all other members
            distances = np.linalg.norm(members - members[-1], axis=1)
            # Keep only those members that are farther than the threshold distance from the last member
            indexes = distances > threshold
            filtered_members = members[indexes]
            filtered_memberships = memberships[indexes]
            filtered_timestamps = timestamps[indexes]
            # Update size of the space
            size = len(filtered_members)
            # If any points were removed, add the last member back to the space and fit the model again to check the score
            if size < previous_size - 1:
                # Add the last member back to the filtered members, memberships and timestamps arrays
                filtered_members = np.concatenate([filtered_members, members[-1:]], axis=0)
                filtered_memberships = np.concatenate([filtered_memberships, memberships[-1:]], axis=0)
                filtered_timestamps = np.concatenate([filtered_timestamps, timestamps[-1:]], axis=0)
                # Reload the data structure with the filtered members and memberships
                self.reload_members(filtered_members, filtered_memberships, timestamps=filtered_timestamps)
                # Fit the model and calculate the score with the filtered members
                if self.learnable():
                    score = self.fit_and_score()
                else:
                    score = 1.0 # Prevent training with insufficient data, which can lead to errors in the SVM.


    def add_point(self, perceptions, confidences):
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
        pos = super().add_point(perceptions, confidences)
        if self.learnable():
            self.fit_and_score()
        prediction = self.get_probability(perceptions)
        if ((confidences > 0.0) and (prediction <= 0.0)) or (
            (confidences <= 0.0) and (prediction > 0.0)
        ):
            if self.fit_and_score() < 1.0:
                self.remove_close_points()
        return pos

    def get_probability(self, perceptions):
        """
        Calculate the new activation value for multiple perception rows.

        :param perceptions: The given perceptions to calculate the activation.
        :type perceptions: core.container.Container
        :return: The activation values, one per perception row.
        :rtype: np.ndarray
        """
        # Obtain the datapoint from the given perception (selects the appropriate features)
        if self._data is not None:
            points = self.data_from_perception(perceptions)
        else:
            return np.zeros(perceptions.size, dtype=float)
        # Calculate the activation value
        if self.learnable():
            output = self.model.decision_function(points)
            activation = np.minimum(np.full_like(output, 2.0), output) / 2.0
        else:
            activation = np.ones_like(points[:, 0])  # Default activation when not learnable (e.g., no points or only antipoints)
        if self.parent_space:
            parent_act = self.parent_space.get_probability(perceptions)
            activation = np.minimum(activation, parent_act)
        return activation


class ANNSpace(PointBasedSpace):
    """
    Use and train a Neural Network to calculate the activations.
    """

    def __init__(self, max_data=2000, sampled_points=200, train_every=1, batch_size=25, epochs=1, output_activation="sigmoid", hidden_activation="relu", hidden_layers=[64, 32], learning_rate=0.05, validation_split=0.0, loss_function=nn.BCEWithLogitsLoss, val_function=nn.BCEWithLogitsLoss, model_file=None, device="cuda", **kwargs):
        
        # Device configuration
        if device not in ["cpu", "cuda"]:
            raise ValueError("Invalid device specified. Use 'cpu' or 'cuda'.")
        elif device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available. Use 'cpu' or ensure CUDA is properly installed.")
        self.device = device
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.max_data = max_data
        self.sampled_points = sampled_points
        self.train_every = train_every
        self.new_points = 0

        # Model and optimizer will be initialized later
        self.configured = False
        self.model_file = model_file
        self.model = None
        self.optimizer = None
        self.criterion = loss_function(reduction="none")
        self.val_criterion = val_function()

        if self.model_file is not None:
            self.load_model()

        super().__init__(**kwargs)

    def configure_model(self, input_length):
        """Configure the ANN model architecture and initialize the optimizer.

        :param input_length: Number of input features for the neural network.
        :type input_length: int
        :param output_length: Number of output features/predictions from the neural network.
        :type output_length: int
        """
        # Seed torch (and python/numpy globals) so weight initialisation and
        # DataLoader shuffling are reproducible when a seed is configured.
        set_global_seeds(self.random_seed)
        self.model = ANNModel_classification(
            input_size=input_length,
            hidden_layers=self.hidden_layers,
            hidden_activation=self.hidden_activation
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.input_length = input_length
        self.configured = True
        
        self.logger.info(f"Model configured with input: {input_length}")

    def load_model(self):
        """Loads model from file."""
        if os.path.exists(self.model_file):
            checkpoint = torch.load(self.model_file, map_location=self.device)
            
            # Extract model configuration from checkpoint
            self.input_length = checkpoint['input_length']
            self.hidden_layers = checkpoint['hidden_layers']
            self.hidden_activation = checkpoint['hidden_activation']
            self.learning_rate = checkpoint['learning_rate']
            
            # Configure model architecture
            self.configure_model(self.input_length)
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            self.logger.info(f"Model loaded from {self.model_file}")
        else:
            self.logger.warning(f"Model file {self.model_file} not found")

    def save_model(self, filepath):
        """Save the trained model to a PyTorch checkpoint file.

        :param filepath: Path where the model checkpoint will be saved. Automatically adds '.pth' extension if not present.
        :type filepath: str
        :return: Tuple containing success status and the path where the model was saved.
        :rtype: tuple(bool, str)
        """        
        if not self.configured:
            self.logger.warning("Model not configured. Cannot save.")
            return False, ""
            
        filepath = filepath if filepath.endswith('.pth') else filepath + '.pth'
        if filepath is None:
            self.logger.warning("No save path provided")
            return False, ""
            
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_length': self.input_length,
            'hidden_layers': self.hidden_layers,
            'hidden_activation': self.hidden_activation,
            'output_activation': self.output_activation,
            'learning_rate': self.learning_rate
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Model saved to {filepath}")
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

    def _train(self, x_train, y_train, epochs=None, batch_size=None, validation_split=0.0, 
              x_val=None, y_val=None, sample_weights=None, verbose=1, reset_optimizer=True):
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
        :param sample_weights: Optional weights for the training samples, defaults to None
        :type sample_weights: np.ndarray or torch.Tensor, optional
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
            sample_weights = sample_weights[:split_idx] if sample_weights is not None else None
            
            val_dataset = TensorDataset(x_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
        # Create data loader for training
        train_dataset = WeightedTensorDataset(x_train, y_train, weights=sample_weights)
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
            
            for batch_x, batch_y, batch_weights in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_weights = batch_weights.to(self.device).float()
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_x)
                per_sample_loss = self.criterion(outputs, batch_y)
                loss = (per_sample_loss * batch_weights.unsqueeze(-1)).mean()
                
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
                        self.logger.info(f"Early stopping at epoch {epoch+1}. Restoring weights from epoch {best_epoch+1} with val loss {best_val_loss:.4f}.")
                    # Restore best weights
                    self.model.load_state_dict(self.best_weights)
                    break
                    
            if verbose > 0:
                if val_loss is not None:
                    self.logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")
                else:
                    self.logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f}")

    def _call(self, x):
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
            predictions = torch.sigmoid(self.model(x)) # TODO: Make activation function configurable
        
        predictions=predictions.squeeze().cpu().numpy()
        return predictions
    
    def get_weights(self):
        """Get the current model , uses the state dictionary."""
        if not self.configured:
            self.logger.warning("Model not configured. Cannot get weights.")
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
            self.logger.warning("Model not configured. Cannot set weights.")
            return False
            
        try:
            self.model.load_state_dict(weights)
            self.logger.info("Weights set successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to set weights: {e}")
            return False
        
    def add_point(self, perceptions, confidences):
        """
        Add a new point to the P-Node.

        :param perceptions: A given perception to add.
        :type perceptions: dict
        :param confidences: The confidences of the added points that specify if they are points or antipoints.
        :type confidences: float
        :return: The position of the added point.
        :rtype: int
        """
        #self.semaphore.acquire()
        pos = super().add_point(perceptions, confidences)
        self.new_points += len(pos) if isinstance(pos, list) else 1

        if self.learnable():
            # If the model is not built yet, build it
            if self.configured == False:
                input_shape = self.members.shape[1]  # Get the number of features from the point
                self.configure_model(input_shape)

            if self.new_points>=self.train_every:
                self.train_step()
                self.new_points = 0 
        return pos

    def train_step(self):
        self.logger.info(f"Training on {self.new_points}")
        if self.size > self.max_data:
            self.logger.info(f"Using last {self.max_data} points for training.")
            first_data = self.size - self.max_data
        else:
            first_data = 0

        members = self.members[first_data : self.size]
        memberships = self.memberships[first_data : self.size].copy()
        memberships[memberships <= 0] = 0.0 # Clamp negative memberships to 0
        members_size = len(members)
        n_samples = min(self.sampled_points, members_size)
        idx = self.rng.choice(members_size, size=n_samples, replace=False)

        X = members[idx]
        Y = memberships[idx]
        n_0 = int(len(Y[Y == 0.0]))
        n_1 = int(len(Y[Y == 1.0]))
        weight_for_0 = (
            (1 / n_0) * (X.shape[0] / 2.0) if n_0 != 0 else 1.0
        )
        weight_for_1 = (
            (1 / n_1) * (X.shape[0] / 2.0) if n_1 != 0 else 1.0
        )
        # This supports the case of points that have lower confidence (between 0 and 1) which are weighted with 1. While points and antipoints are balanced.
        weights = np.ones_like(Y)
        weights[Y == 0.0] = weight_for_0
        weights[Y == 1.0] = weight_for_1

        self.logger.info(f"Training data distribution: Total: {len(Y)}, 0s={n_0}, 1s={n_1}, weights: 0={weight_for_0}, 1={weight_for_1}")
        self._train(X, Y, validation_split=self.validation_split, sample_weights=weights, reset_optimizer=False)

    def get_probability(self, perceptions):
        """
        Calculate the new activation value for multiple perception rows.

        :param perceptions: The given perceptions to calculate the activation.
        :type perceptions: core.container.Container
        :return: The activation values, one per perception row.
        :rtype: np.ndarray
        """
        # Obtain the datapoint from the given perception (selects the appropriate features)
        if self._data is not None:
            points = self.data_from_perception(perceptions)
        else:
            return np.zeros(perceptions.size, dtype=float)
        # Calculate the activation value
        if self.configured:
            activation = self._call(points)
        else:
            activation = np.ones_like(points[:, 0], dtype=float)  # Default to 1.0 if model is not configured
        if self.parent_space:
            parent_act = self.parent_space.get_probability(perceptions)
            activation = np.minimum(activation, parent_act)
        return activation

        
    
        
class ANNModel_classification(nn.Module):
    """PyTorch neural network model."""
    
    def __init__(self, input_size, hidden_layers=[128], 
                 hidden_activation='relu'):
        """Initialize the PyTorch neural network model with configurable architecture.

        :param input_size: Number of input features to the network.
        :type input_size: int
        :param hidden_layers: List of integers specifying the size of each hidden layer, defaults to [128]
        :type hidden_layers: list, optional
        :param hidden_activation: Activation function for hidden layers ('relu', 'tanh', 'sigmoid'), defaults to 'relu'
        :type hidden_activation: str, optional
        """        
        super(ANNModel_classification, self).__init__()
        
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
        self.layers.append(nn.Linear(prev_size, 1))
        
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


class WeightedTensorDataset(TensorDataset):
    """TensorDataset that includes sample weights, defaults to ones if not provided."""
    def __init__(self, x, y, weights=None):
        super().__init__(x, y)
        n = len(x)
        if weights is None:
            weights = torch.ones(n)
        # convert and normalize shape to 1D
        if not isinstance(weights, torch.Tensor):
            weights = torch.FloatTensor(weights)
        else:
            weights = weights.float()
        if weights.dim() == 2 and weights.shape[1] == 1:
            weights = weights.squeeze(1)
        if weights.shape[0] != n:
            raise ValueError(f"weights length ({weights.shape[0]}) must match number of samples ({n})")
        self.weights = weights
        
    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        w = self.weights[idx]
        return x, y, w
