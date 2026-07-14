This is part of the e-MDB architecture documentation. Main page [here.](https://docs.pillar-robots.eu/en/latest/)

# e-MDB Cognitive Nodes implemented by the GII

This [repository](https://github.com/pillar-robots/emdb_cognitive_nodes_gii) contains the packages that implement the reference cognitive nodes of the software implementation of the e-MDB cognitive architecture developed under the [PILLAR Robots project](https://pillar-robots.eu/).

There are two ROS 2 packages in this repository:

- **cognitive_nodes:** Implementations of the cognitive nodes.
- **cognitive_nodes_interfaces:** Services and messages definitions.

We can find four sections in this documentation:

- [Concepts:](concepts/concepts.md) Theoretical concepts about the cognitive nodes into the cognitive architecture and explanation of the intrinsic motivations implemented in the cognitive architecture.
- [Cognitive Nodes API documentation:](cognitive_nodes/cn_api_documentation.rst) API of the cognitive nodes implemented by the GII.
- [Intrinsic Motivations API documentation:](cognitive_nodes/im_api_documentation.rst) API of the intrinsic motivations implemented by the GII.
- [Alignment API documentation:](cognitive_nodes/alignment_api_documentation.rst) API of the alignment process implemented by the GII.
- [Auxiliary Scripts API documentation:](cognitive_nodes/as_api_documentation.rst) API of the auxiliary scripts used in the cognitive nodes.

```{toctree}
:caption: e-MDB Cognitive Nodes
:hidden:
:glob:

concepts/*

```

```{toctree}
:caption: API Documentation
:hidden:
:glob:

cognitive_nodes/cn_api_documentation.rst
cognitive_nodes/im_api_documentation.rst
cognitive_nodes/as_api_documentation.rst
cognitive_nodes/alignment_api_documentation.rst

```
