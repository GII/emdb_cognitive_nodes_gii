=================
API Documentation
=================

++++++++++
Perception
++++++++++

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

**Specific services**

/pnode/id/add_point => Add a point (or anti-point) to the underlying space.

.. automodule:: cognitive_nodes.pnode
    :members:
    :show-inheritance:

++++++++++
Space
++++++++++

.. automodule:: cognitive_nodes.space
    :members:
    :show-inheritance:


++++++++++
Need
++++++++++

**WORK IN PROGESS**

.. automodule:: cognitive_nodes.need
    :members:
    :show-inheritance:

++++++++++
Drive
++++++++++

**WORK IN PROGESS**

.. automodule:: cognitive_nodes.drive
    :members:
    :show-inheritance:

++++++++++
Goal
++++++++++

**Specific services**

/goal/id/set_activation => Drives can modify a goal’s activation.

/goal/id/get_reward => Obtains the reward value.

/goal/id/is_reached => Check if the goal has been reached.


.. automodule:: cognitive_nodes.goal
    :members:
    :show-inheritance:

++++++++++++
World model
++++++++++++

**Specific services**

/world_model/id/set_activation => Some processes (ie: the main cognitive loop) can
modify a world model’s activation.

.. automodule:: cognitive_nodes.world_model
    :members:
    :show-inheritance:

++++++++++++
Utiliy model
++++++++++++

**WORK IN PROGRESS**

.. automodule:: cognitive_nodes.utility_model
    :members:
    :show-inheritance:

++++++++++++
Policy
++++++++++++

**Specific services**

/policy/id/set_activation => C-nodes can modify a policy’s activation.

/policy/id/execute => Execute the policy.


.. automodule:: cognitive_nodes.policy
    :members:
    :show-inheritance:

++++++++++
C-Node
++++++++++

.. automodule:: cognitive_nodes.cnode
    :members:
    :show-inheritance: