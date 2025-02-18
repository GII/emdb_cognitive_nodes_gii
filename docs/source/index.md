This is part of the e-MDB architecture documentation. Main page [here.](https://docs.pillar-robots.eu/en/latest/)

# e-MDB Cognitive Nodes implemented by the GII

This [repository](https://github.com/pillar-robots/emdb_cognitive_nodes_gii) contains the packages that implement the reference cognitive nodes of the software implementation of the e-MDB cognitive architecture developed under the [PILLAR Robots project](https://pillar-robots.eu/).
It includes implementations for Perceptions, P-Nodes, Goals, World Models, Utility Models, C-Nodes, Needs, Drives, and Policies, although some of them are barebones at the moment.

There are two ROS 2 packages in this repository:

- **cognitive_nodes:** Implementations of the cognitive nodes.
- **cognitive_nodes_interfaces:** Services and messages definitions.

We can find two sections in this documentation:

- [Concepts:](concepts/concepts.md) Theorical concepts about the cognitive nodes into the cognitive architecture.
- [API documentation:](cognitive_nodes/api_documentation.rst) API of the cognitive nodes implemented by the GII.  

```{toctree}
:caption: e-MDB Cognitive Nodes
:hidden:
:glob:

concepts/*
cognitive_nodes/*

```
