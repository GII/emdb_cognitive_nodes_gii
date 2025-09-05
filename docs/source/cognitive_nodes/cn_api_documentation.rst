=================================
Cognitive Nodes
=================================

Here you can find a description of all the scripts to implement the different cognitive nodes, 
their specific topics and services, and the documentation of their methods.

The API reference of the base class CognitiveNode can be found in the `core API Reference <https://docs.pillar-robots.eu/projects/emdb_core/en/latest/core/API.html#cognitive-node>`_

++++++++++
Perception
++++++++++

Python script to implement the Perception cognitive node, which is responsible for processing sensory inputs and generating perceptions.

**Specific topics**

/perception/id/value => Anytime a new perception is generated, it is published here.

**Specific services**

/perception/id/set_activation => Attention mechanisms can modify a perception's activation.

.. automodule:: cognitive_nodes.perception
    :members:
    :show-inheritance:


++++++++++
P-Node
++++++++++

Python script to implement the P-Node cognitive node, which represent perceptual equivalence classes (discretizations of continuous perceptual space) 
used to operationally group perceptions that lead to the same outcome under identical actions.

**Spacific topics**

/pnode/id/success_rate => The success rate (percentage of new points that belong to the space) of the P-Node is published here.

**Specific services**

/pnode/id/add_point => Add a point (or anti-point) to the underlying space.

/pnode/id/send_space => Send the underlying space to the client.

/pnode/id/contais_space => Check if the underlying space contais a given space.

.. automodule:: cognitive_nodes.pnode
    :members:
    :show-inheritance:


++++++++++
Need
++++++++++

Python script to implement the Needs and Missions of the cognitive architecture, which represent the desired motivational state of the system.
Needs represent intrinsic motivations, while Missions represent extrinsic motivations.

**Specific services**

/need/id/set_activation => Purposes can modify a need's or mission's activation.

/need/id/get_satisfaction => Obtain the satisfaction value of a need (1.0 if satisfied, 0.0 if not).

.. automodule:: cognitive_nodes.need
    :members:
    :show-inheritance:

++++++++++
Drive
++++++++++

Python script to implement the Drives, which are defined as a fuction that provides a measure (evaluation) of how desirable the satisfaction of a motivational desire (Need or Mission) is.

**Specific topics**

/drive/id/evaluation => Publish the evaluation of the drive.

**Specific services**

/drive/id/set_activation => Needs and missions can modify a drive's activation.

/drive/id/get_sucess_rate => Get a prediction success rate based on a historic of previous predictions.

/drive/id/get_reward => Obtain the reward value. A Drive gives reward if its evaluation value decreases.

/drive/id/get_effects => In case of DriveEffectanceExternal, this service returns the effects that were found in the environment.

/drive/id/get_knowledge => In case of ProspectionDrive, this service returns the knowledge that was found.

.. automodule:: cognitive_nodes.drive
    :members:
    :show-inheritance:

++++++++++
Goal
++++++++++

Python script to implement the Goal cognitive node, which represents an area in the state space that, when reached, lead to the reduction of at least one
of the drives that are part of the robot's motivational system. That is, it is implicitly a rewarded area.

**Spacific topics**

/goal/id/confidence => The learning confidence of the goal's space, if exists, is published here, measured as the success rate of the predictions of received rewards.

**Specific services**

/goal/id/set_activation => Drives can modify a goal's activation.

/goal/id/get_reward => Obtains the reward value.

/goal/id/is_reached => Check if the goal has been reached.

/goal/id/send_space => Send the underlying space to the client, if exists.

/goal/id/contains_space => Check if the underlying space contains a given space, if the goal has one.

.. automodule:: cognitive_nodes.goal
    :members:
    :show-inheritance:

++++++++++++
World model
++++++++++++

(**WORK IN PROGRESS**)

Python script to implement the World Model cognitive node, which is represent the behavior of the domain in which the robot is operating.
They are usually instantiated as a predictor of the perceptual situation Pt+1 that will result from the application of an action when in 
a perceptual state Pt.

**Specific services**

/world_model/id/set_activation => Some processes (ie: the main cognitive loop) can
modify a world model's activation.

/world_model/id/predict => Get predicted perception values for the last perceptions not newer than a given
timestamp and for a given policy.

/world_model/id/get_success_rate => Get a prediction success rate based on a historic of previous predictions.

/world_model/id/is_compatible => Check if the Model is compatible with the current available perceptions.



.. automodule:: cognitive_nodes.world_model
    :members:
    :show-inheritance:

++++++++++++
Utiliy model
++++++++++++

(**WORK IN PROGRESS**)

Python script to implement the Utility Model cognitive node, which estimates the expected utility of perceptual states with respect to a goal, 
based on the probability of achieving it and the potential reward.

**Specific services**

/utility_model/id/set_activation => C-Nodes can modify a utility model's activation.

/utility_model/id/predict => Get predicted perception values for the last perceptions not newer than a given
timestamp and for a given policy.

/utility_model/id/get_success_rate => Get a prediction success rate based on a historic of previous predictions.

/utility_model/id/is_compatible => Check if the Model is compatible with the current available perceptions.

.. automodule:: cognitive_nodes.utility_model
    :members:
    :show-inheritance:

++++++++++++
Policy
++++++++++++

Python script to implement the Policy cognitive node, which is a reactive decision structure in the form of a procedural componen that provides the 
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

Python script to implement the C-Node cognitive node, which represens a context. It links a P-Node (initial state), the World Model and a Goal (desired state), 
with the Policy needed to move from the initial to the desired state.

.. automodule:: cognitive_nodes.cnode
    :members:
    :show-inheritance: