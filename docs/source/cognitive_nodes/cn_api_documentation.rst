=================================
Cognitive Nodes
=================================

Here you can find a description of all the scripts to implement the different cognitive nodes, 
their specific topics and services, and the documentation of their methods.

The API reference of the base class CognitiveNode can be found in the `core API Reference <https://docs.pillar-robots.eu/projects/emdb_core/en/latest/core/API.html#cognitive-node>`_

++++++++++
Perception
++++++++++

Python module that implements the Perception cognitive node, which is responsible for processing sensory inputs and generating perceptions.

**Specific topics**

/perception/id/value => Anytime a new perception is generated, it is published here, as a core_interfaces/Container message.

**Specific services**

/perception/id/set_activation => Attention mechanisms can modify a perception's activation.

/perception/id/set_inputs => Set the inputs of the perception (not implemented yet).

.. automodule:: cognitive_nodes.perception
    :members:
    :show-inheritance:


++++++++++
P-Node
++++++++++

Python module that implements the P-Node cognitive node, which represent perceptual equivalence classes (discretizations of continuous perceptual space) 
used to operationally group perceptions that lead to the same outcome under identical actions.

Each P-Node caches the activations of its space by the values of the perception, so a perception that has not changed does not require evaluating the space again. The cache is emptied when the space changes.

**Specific topics**

/pnode/id/success_rate => The success rate (percentage of new points that belong to the space) of the P-Node is published here.

**Specific services**

/pnode/id/add_points => Add points (or anti-points) to the underlying space.

/pnode/id/send_space => Send the underlying space to the client.

/pnode/id/contains_space => Check if the underlying space contains a given space.

/pnode/id/load_space => Load a complete space into the P-Node.

/pnode/id/save_model => Save the model of the underlying space to a file.

.. automodule:: cognitive_nodes.pnode
    :members:
    :show-inheritance:


++++++++++++
RobotPurpose
++++++++++++

Python module that implements the Needs and Missions of the cognitive architecture, which represent the desired motivational state of the system.
Needs represent intrinsic motivations, while Missions represent extrinsic motivations.

**Specific services**

/robot_purpose/id/set_activation => Service to set the activation of the robot purpose.

/robot_purpose/id/get_satisfaction => Obtain the satisfaction value of a need (1.0 if satisfied, 0.0 if not).

.. automodule:: cognitive_nodes.robot_purpose
    :members:
    :show-inheritance:

++++++++++
Drive
++++++++++

Python module that implements the Drives, which are defined as a function that provides a measure (evaluation) of how desirable the satisfaction of a motivational desire (Need or Mission) is.

**Specific topics**

/drive/id/evaluation => Publish the evaluation of the drive.

**Specific services**

/drive/id/set_activation => Needs and missions can modify a drive's activation.

/drive/id/get_success_rate => Get a prediction success rate based on a historic of previous predictions (not implemented yet).

/drive/id/get_reward => Obtain the reward value. A Drive gives reward if its evaluation value decreases.

/drive/id/get_effects => In case of DriveEffectanceExternal, this service returns the effects that were found in the environment.

/drive/id/get_knowledge => In case of ProspectionDrive, this service returns the knowledge that was found.

/drive/id/get_reusable_knowledge => In case of DriveKnowledgeReuse, this service returns the knowledge that can be reused.

.. automodule:: cognitive_nodes.drive
    :members:
    :show-inheritance:

++++++++++
Goal
++++++++++

Python module that implements the Goal cognitive node, which represents an area in the state space that, when reached, lead to the reduction of at least one
of the drives that are part of the robot's motivational system. That is, it is implicitly a rewarded area.

**Specific topics**

/goal/id/confidence => The learning confidence of the goal's space, if exists, is published here, measured as the success rate of the predictions of received rewards.

**Specific services**

/goal/id/set_activation => Drives can modify a goal's activation.

/goal/id/get_reward => Obtains the reward value.

/goal/id/is_reached => Check if the goal has been reached.

/goal/id/add_points => Add points (or anti-points) to the underlying space, if exists.

/goal/id/send_space => Send the underlying space to the client, if exists.

/goal/id/contains_space => Check if the underlying space contains a given space, if the goal has one.

.. automodule:: cognitive_nodes.goal
    :members:
    :show-inheritance:

++++++++
Episodes
++++++++

Provides a data structure to represent episodes. It also provides methods for its manipulation and conversion between different formats.

.. automodule:: cognitive_nodes.episode
    :members:
    :show-inheritance:


+++++++++++++++
Episodic Buffer
+++++++++++++++

Python module that implements episodic buffers of different types. These can be instantiated and used by other cognitive nodes to store their short-term memories.
It provides methods for the manipulation of the buffer and its contents.

.. automodule:: cognitive_nodes.episodic_buffer
    :members:
    :show-inheritance:

++++++++++++++++++
Deliberative model
++++++++++++++++++

Python module that implements the Deliberative Model cognitive node, which is a higher-level cognitive node that integrates the functionalities of World Model, Utility Model. 

**Specific services**

The services of the World Models and the Utility Models start with world_model and utility_model instead of deliberative_model.

/deliberative_model/id/set_activation => C-nodes can modify a deliberative model's activation.

/deliberative_model/id/predict => Get predicted values for the input episodes (a core_interfaces/Container).

/deliberative_model/id/get_success_rate => Get a prediction success rate based on a historic of previous predictions (not implemented yet).

/deliberative_model/id/is_compatible => Check if the Model is compatible with the current available episode (not implemented yet).

/deliberative_model/id/save_model => Save the model to a file if compatible.


.. automodule:: cognitive_nodes.deliberative_model
    :members:
    :show-inheritance:


++++++++++++
World model
++++++++++++


Python module that implements the World Model cognitive node, which is represent the behavior of the domain in which the robot is operating.
They are usually instantiated as a predictor of the perceptual situation Pt+1 that will result from the application of an action when in 
a perceptual state Pt. It inherits all the functionalities of the Deliberative Model, including its services

**Specific topics**

/world_model/id/prediction_error => The evaluator of the World Model publishes its prediction error here.

.. automodule:: cognitive_nodes.world_model
    :members:
    :show-inheritance:

+++++++++++++
Utility model
+++++++++++++

Python module that implements the Utility Model cognitive node, which estimates the expected utility of perceptual states with respect to a goal, 
based on the probability of achieving it and the potential reward.  It inherits all the functionalities of the Deliberative Model, including its services

**Specific services**

/utility_model/id/execute => Executes the deliberation process with this utility model and its associated world model.

/utility_model/id/add_trace => In case of LearnedUtilityModel, adds a trace of episodes to the model.

.. automodule:: cognitive_nodes.utility_model
    :members:
    :show-inheritance:

++++++++++++
Policy
++++++++++++

Python module that implements the Policy cognitive node, which is a reactive decision structure in the form of a procedural component that provides the 
action to apply when at a given perceptual point.

**Specific services**

/policy/id/set_activation => C-nodes can modify a policy's activation.

/policy/id/execute => Execute the policy.


.. automodule:: cognitive_nodes.policy
    :members:
    :show-inheritance:

++++++++++
C-Node
++++++++++

Python module that implements the C-Node cognitive node, which represents a context. It links a P-Node (initial state), the World Model and a Goal (desired state), 
with the Policy needed to move from the initial to the desired state.

**Specific services**

/cnode/id/log_execution => Logs whether an execution of the C-Node was successful.

.. automodule:: cognitive_nodes.cnode
    :members:
    :show-inheritance: