import numpy
import rclpy
from core.cognitive_node import CognitiveNode
from core.service_client import ServiceClient, ServiceClientAsync
from cognitive_node_interfaces.srv import GetActivation
from core.utils import perception_dict_to_msg


class CNode(CognitiveNode):
    """
    CNode class
    It represents a context, that is, a link between nodes that were activated together in the past.
    It is assumed that there is only one element of each type connected to the C-Node.
    """

    def __init__(self, name="cnode", class_name="cognitive_nodes.cnode.CNode", **params):
        """
        Constructor of the CNode class
        Initializes a CNode with the given name and registers it in the LTM

        :param name: The name of the CNode
        :type name: str
        :param class_name: The name of the CNode class
        :type str
        """
        super().__init__(name, class_name, **params)
        self.activation_sources=['WorldModel', 'Goal', 'PNode']
        self.configure_activation_inputs(self.neighbors)

    async def calculate_activation(self, perception=None, activation_list=None):
        """
        Calculate the new activation value by multiplying the activation values of its previous neighbors.
        By default, with percerception = None, it will multiply the last activations of its neighbors, but
        it's possible to use an arbitrary perception, that will propagate to the neighbors, calculating the
        final activation of the CNode for that perception.

        :param perception: Arbitrary perception
        :type perception: dict
        :return: The activation of the CNode
        :rtype: float
        """
        if activation_list==None:
            node_activations = []
            neighbors_name = [
                neighbor["name"] for neighbor in self.neighbors if neighbor["node_type"] != "Policy"
            ]
            for name in neighbors_name:
                perception_msg = perception_dict_to_msg(perception)
                service_name = "cognitive_node/" + str(name) + "/get_activation"
                if not service_name in self.node_clients:
                    self.node_clients[service_name] = ServiceClientAsync(
                        self, GetActivation, service_name, self.cbgroup_client
                    )
                activation = await self.node_clients[service_name].send_request_async(
                    perception=perception_msg
                )
                self.get_logger().debug(f"DEBUG CNODE: Activation for {name}: {activation.activation}")
                node_activations.append(activation.activation)
            self.get_logger().debug(f"DEBUG CNODE: Activation list {node_activations}")
            activation_list = numpy.prod(node_activations)
            self.activation.activation = float(numpy.max(activation_list))
            self.activation.timestamp=self.get_clock().now().to_msg()
            # TODO: Selection of the perception that have the max CNode or PNode activation (if it exists), as in the old MDB

            self.get_logger().debug(
                self.node_type + " activation for " + self.name + " = " + str(self.activation)
            )
        else:
            self.calculate_activation_prod(activation_list)

        return self.activation


def main(args=None):
    rclpy.init(args=args)

    cnode = CNode()

    rclpy.spin(cnode)

    cnode.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
