# e-MDB reference implementation for cognitive nodes

Note:

***WORK IN PROGRESS***

The original repository has been split in 5 and we are refactoring everything, please, be patient while we move and rename everything.

These are the cognitive architecture repositories for PILLAR and their content:

- _emdb_core_. Essential elements of the cognitive architecture. These are necessary to run an experiment using the cognitive architecture.
- _emdb_cognitive_nodes_gii_. Reference implementation for the main cognitive nodes.
- _emdb_cognitive_processes_gii_. Reference implementation for the main cognitive processes.
- _emdb_discrete_event_simulator_gii_. Implementation of a discrete event simulator used in many experiments.
- _emdb_experiments_gii_. Configuration files for experiments.

Current reference implementation for the main cognitive nodes. It includes implementations for P-nodes, goals, world models, utility models, C-nodes, needs, drives, and policies, although some of them are barebones at the moment.

There are two ROS 2 packages in this repository:

- _cognitive_nodes_. Implementations of the cognitive nodes.
- _cognitive_nodes_interfaces_. Services and messages definitions.
