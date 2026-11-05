=====================
Intrinsic Motivations
=====================

Here you can find a description of all the scripts that implement the different intrinsic motivation components,
which drive autonomous exploration and learning in the cognitive architecture.

++++++++++
Novelty
++++++++++

Python script which implements components for exploration intrinsic motivation, based on novelty, which enables
robots to explore their environment.

This module implements the random exploration (PolicyNovelty). The novelty-based exploration is implemented by NoveltyUtilityModel, in the Utility Model module.

.. automodule:: cognitive_nodes.novelty
    :members:
    :show-inheritance:

+++++++++++++++
LLM Exploration
+++++++++++++++

Python script which implements components for exploration intrinsic motivation, guided by a Large Language Model (LLM),
which enables robots to explore their environment with the help of an LLM.

.. note::

    The API of this module is not available in this version of the documentation.

+++++++++++++++
Effectance
+++++++++++++++

Python script which implements components for effectance intrinsic motivation, based on detecting and reproducing effects, which enables
robots to learn from the consequences of their actions in the environment.

This module includes both internal effectance (detecting P-Node consolidation) and external effectance (detecting and reproducing changes 
in sensor values).


.. automodule:: cognitive_nodes.effectance
    :members:
    :show-inheritance:

+++++++++++++++
Prospection
+++++++++++++++

Python script which implements components for prospection intrinsic motivation, focused on discovering hierarchical 
relationships between goals in the system. This enables robots to build knowledge chains by identifying when achieving 
one goal can lead to another.

.. automodule:: cognitive_nodes.prospection
    :members:
    :show-inheritance:


+++++++++++++++
Model creation
+++++++++++++++

Python module that implements components for autonomous model creation, allowing the cognitive architecture to
dynamically generate and integrate new world and utility models based on its interactions with the environment.

.. automodule:: cognitive_nodes.model_creation
    :members:
    :show-inheritance:

+++++++++++++++
Knowledge reuse
+++++++++++++++

Python module that implements components for knowledge reuse intrinsic motivation, which reuses chains of nodes learned in other situations for a newly observed goal.
The drive finds the chains that can be reused, and the policy duplicates and reconnects their nodes.

.. automodule:: cognitive_nodes.knowledge_reuse
    :members: DriveKnowledgeReuse, PolicyKnowledgeReuse
    :show-inheritance:

++++++++++
Competence
++++++++++

Python module that implements the competence intrinsic motivation, which tracks the competence of the robot in achieving its goals and drives it to improve them.

.. automodule:: cognitive_nodes.competence
    :members:
    :show-inheritance:
