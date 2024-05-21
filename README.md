# e-MDB reference implementation for cognitive nodes

This is the repository that contains the packages that implement the reference cognitive nodes of the software implementation of the e-MDB cognitive architecture developed under the [PILLAR Robots project](https://pillar-robots.eu/).
It includes implementations for P-nodes, goals, world models, utility models, C-nodes, needs, drives, and policies, although some of them are barebones at the moment.

There are two ROS 2 packages in this repository:

- _cognitive_nodes_. Implementations of the cognitive nodes.
- _cognitive_nodes_interfaces_. Services and messages definitions.

For more information about the cognitive architecture design, you can visit the [emdb_core](https://github.com/GII/emdb_core?tab=readme-ov-file#design) repository or the [PILLAR Robots official website](https://pillar-robots.eu/).

## Table of Contents

- **[Dependencies](#dependencies)**
- **[Installation](#installation)**
- **[Configurate an experiment](#configurate-an-experiment)**
- **[Execution](#execution)**
- **[Results](#results)**

## Dependencies

These are the dependencies required to use this repository of the e-MDB cognitive architecture software:

- ROS 2 Humble
- Numpy 1.24.3
- Sklearn 1.4.2
- Tensorflow 2.15.0
  
Other versions could work, but the indicated ones have proven to be functional.

## Installation

To install this package, it's necessary to clone this repository in a ROS workspace and build it with colcon.

```
colcon build --symlink-install
```
This respository only constitutes the reference cognitive nodes of the e-MDB cognitive architecture. To get full functionality, it's required to add to the ROS workspace, at least, the [emdb_core](https://github.com/GII/emdb_core) repository, that constitutes the base of the architecture, and other packages that include the cognitive processes, the experiment configuration and the interface that connects the architecture with a real or a simulated environment. Therefore, to use the first version of the architecture implemented by GII, these repositories need to be cloned into the workspace:
- [_emdb_core_]([https://github.com/GII/emdb_cognitive_nodes_gii](https://github.com/GII/emdb_core)). Core of the cognitive architecture.
- [_emdb_cognitive_processes_gii_](https://github.com/GII/emdb_cognitive_processes_gii). Reference implementation for the main cognitive processes.
- [_emdb_discrete_event_simulator_gii_](https://github.com/GII/emdb_discrete_event_simulator_gii). Implementation of a discrete event simulator used in many experiments.
- [_emdb_experiments_gii_](https://github.com/GII/emdb_experiments_gii). Configuration files for experiments.

In these repositories is included an example experiment with the discrete event simulator in which the Policies, the Goal and the World Model are defined in the beginning, the objective being to create the corresponding PNodes and CNodes, which allow the Goal to be achieved effectively by the simulated robot. 

The Goal, called ObjectInBoxStandalone, consists of introducing a cylinder into a box correctly. For that, the robot can use, in a World Model called GripperAndLowFriction, the following policies:
- Grasp object: use one of the two grippers to grasp an object
- Grasp object with two hands: use both arms to grasp an object between their ends
- Change hands: move object from one gripper to the other 
- Sweep object: sweep an object to the central line of the table
- Ask nicely: ask experimenter, simulated in this case, to bring something to within reach
- Put object with robot: deposit an object to close to the robot base
- Put object in box: place an object in a receptacle
- Throw: throw an object to a position
  
The reward obtained could be 0.2, 0.3 or 0.6 if the robot with its action improves its situation to get the final goal. Finally, when the cylinder is introduced into the box, the reward obtained is 1.0. Thus, at the end of the experiment, seven PNodes and CNodes should be created, one per policy, except Put object with robot, which doesn't lead to any reward.

## Configurate an experiment

There is a general and basic implementation of each cognitive node, which constitutes its parent class. It's possible to create customized cognitive nodes for a specific application creating a child class of them. For instance, in the example experiment, there is a child class in Goal (GoalObjectInBoxStandalone) and Perception (DiscreteEventSimulatorPerception) cognitive nodes, to read the perceptions and get the reward from the discrete event simulator.

Also, it's important to talk about Space class, contained in space.py. This class is used by PNodes and sometimes Goals to store perceptions and calculate their activation. In the example experiment, the Goal implemented doesn´t use the Space class, but PNodes have some implementations to try: it's possible to calculate their activation by using Artificial Neural Networks (ANNs), SVM or Point Based Rules.

To configure all this stuff, it's necessary to edit the experiment configuration file, stored in the [_emdb_experiments_gii_](https://github.com/GII/emdb_experiments_gii) repository (experiments/default_experiment.yaml) or in an experiments package created by oneself.

There we can indicate the nodes that have to be created when the experiment is launched. For instance, as has been said before, Policies are predefined in the example experiment, so we have to specify their parameters in the configuration file:
```
Nodes:

        Policy:
            -
                name: grasp_object
                class_name: cognitive_nodes.policy.Policy
                parameters:
                    publisher_msg: std_msgs.msg.String
                    publisher_topic: /mdb/baxter/executed_policy 
            -
                name: grasp_with_two_hands
                class_name: cognitive_nodes.policy.Policy
                parameters:
                    publisher_msg: std_msgs.msg.String
                    publisher_topic: /mdb/baxter/executed_policy  
            -
                name: change_hands
                class_name: cognitive_nodes.policy.Policy
                parameters:
                    publisher_msg: std_msgs.msg.String
                    publisher_topic: /mdb/baxter/executed_policy 
            -
                name: sweep_object
                class_name: cognitive_nodes.policy.Policy
                parameters:
                    publisher_msg: std_msgs.msg.String
                    publisher_topic: /mdb/baxter/executed_policy 
            -
                name: put_object_in_box
                class_name: cognitive_nodes.policy.Policy
                parameters:
                    publisher_msg: std_msgs.msg.String
                    publisher_topic: /mdb/baxter/executed_policy  
            -
                name: put_object_with_robot
                class_name: cognitive_nodes.policy.Policy
                parameters:
                    publisher_msg: std_msgs.msg.String
                    publisher_topic: /mdb/baxter/executed_policy 
            -
                name: throw
                class_name: cognitive_nodes.policy.Policy
                parameters:
                    publisher_msg: std_msgs.msg.String
                    publisher_topic: /mdb/baxter/executed_policy 
            -
                name: ask_nicely
                class_name: cognitive_nodes.policy.Policy
                parameters:
                    publisher_msg: std_msgs.msg.String
                    publisher_topic: /mdb/baxter/executed_policy
```
Also, the cognitive architecture can create nodes while is running, as is the case of the PNodes and CNodes in the example. Therefore, it's necessary to indicate the default class that the new nodes are going to use:
```
Connectors:
        -
            data: Space
            default_class: cognitive_nodes.space.SVMSpace
        -
            data: Perception
            default_class: cognitive_nodes.perception.Perception
        -
            data: PNode
            default_class: cognitive_nodes.pnode.PNode
        -
            data: CNode
            default_class: cognitive_nodes.cnode.CNode
        -
            data: Goal
            default_class: cognitive_nodes.goal.Goal
        -
            data: WorldModel
            default_class: cognitive_nodes.world_model.WorldModel
        -
            data: Policy
            default_class: cognitive_nodes.policy.Policy
```
The parameters of the cognitive nodes created during execution are indicated by their corresponding cognitive process.

## Execution

To execute the example experiment or another launch file, it's essential to source the ROS workspace:
```
source install/setup.bash
```
Afterwards, the experiment can be launched:
```
ros2 launch core example_launch.py
```
Once executed, it is possible to see the logs in the terminal, being able to follow the behavior of the experiment in real time.

## Results

Executing the example experiment, it will create two files by default: goodness.txt and pnodes_success.txt. 

In the first one, it is possible to observe important information, such as the policy executed and the reward obtained per iteration. It is possible to observe the learning process by seeing this file in real time with the following command:
```
tail -f goodness.txt
```
In the second file, it's possible to see the activation of the PNodes and if it was a point (True) or an anti-point (False).

When the execution is finished, it's possible to obtain statistics about reward and PNodes activations per 100 iterations by using the scripts available in the scripts directory of the core package (emdb_core/core/scripts):
```
python3 $ROS_WORKSPACE/src/emdb_core/core/scripts/generate_grouped_statistics -n 100 -f goodness.txt > goodness_grouped_statistics.csv

python3 $ROS_WORKSPACE/src/emdb_core/core/scripts/generate_grouped_success_statistics -n 100 -f pnodes_success.txt > pnodes_grouped_statistics.csv
```
To use these scripts it's necessary to have installed python-magic 0.4.27 dependency.

By plotting the data of these final files, it is possible to obtain a visual interpretation of the learning of the cognitive architecture.
