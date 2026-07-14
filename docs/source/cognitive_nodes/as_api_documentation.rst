==================
Auxiliary Scripts
==================

Here you can find a description of all the scripts that provide supporting functionality for the cognitive nodes.
These auxiliary modules implement common tools and abstractions which are used in this type of nodes.

++++++++++
Space
++++++++++

Python script to implement a space used by P-Nodes and Goals, which discretizes a continuous perceptual domain. 
Different types of spaces have been implemented, depending on the system used to determine whether a new perception belongs to an existing perceptual class:

- **PointBased Space**: Uses rules that relate the position of the new perceptionto the closest old ones contained in the space.
- **SVM Space**: Uses Support Vector Machines (SVMs), with Scikit-learn.
- **ANN Space**: Uses Artificial Neural Networks (ANNs), with Tensorflow.

.. automodule:: cognitive_nodes.space
    :members:
    :show-inheritance:


++++++++++
Utils
++++++++++

Python script which implements auxiliary functions used into the scripts of the cognitive nodes.

.. automodule:: cognitive_nodes.utils
    :members:
    :show-inheritance: