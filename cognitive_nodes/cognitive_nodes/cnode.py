import numpy
import rclpy
from core.cognitive_node import CognitiveNode
from core.service_client import ServiceClientAsync
from cognitive_node_interfaces.srv import GetActivation
from core.utils import perception_dict_to_msg


class CNode(CognitiveNode):
    """
    C-Node class.
    It represents a context, i.e. a link between nodes that were activated together in the past.
    It is assumed that there is only one element of each type connected to the C-Node.
    """

    def __init__(self, name: str = "cnode",
                 class_name: str = "cognitive_nodes.cnode.CNode",
                 **params):
        """
        Constructor of the C-Node class.
        Initializes a C-Node with the given name and registers it in the LTM.
        """
        super().__init__(name, class_name, **params)
        # Configure which neighbors provide activation inputs
        self.configure_activation_inputs(self.neighbors)

    async def calculate_activation(self, perception=None, activation_list=None):
        """
        Calculate the new activation value.

        If activation_list is None:
          - Request the activation of all neighbors (except Policies) for the given perception.
          - Compute the product of their activations.
        If activation_list is provided:
          - Delegate to calculate_activation_prod to use the cached activations.

        :param perception: Arbitrary perception to propagate to neighbors.
        :type perception: dict
        :param activation_list: Dictionary with the activation of multiple nodes.
        :type activation_list: dict
        :return: The activation of the C-Node and its timestamp.
        :rtype: cognitive_node_interfaces.msg.Activation
        """
        if activation_list is None:
            node_activations = []
            neighbors_name = [
                neighbor["name"]
                for neighbor in self.neighbors
                if neighbor["node_type"] != "Policy"
            ]
            
            self.get_logger().info(f"CNODE_ACTIVATION - {self.name}: Calculating from {len(neighbors_name)} neighbors: {neighbors_name}")

            for name in neighbors_name:
                perception_msg = perception_dict_to_msg(perception)
                service_name = f"cognitive_node/{name}/get_activation"

                if service_name not in self.node_clients:
                    self.node_clients[service_name] = ServiceClientAsync(
                        self, GetActivation, service_name, self.cbgroup_client
                    )

                activation = await self.node_clients[service_name].send_request_async(
                    perception=perception_msg
                )
                self.get_logger().info(
                    f"CNODE_ACTIVATION - {self.name}: Neighbor {name} returned activation={activation.activation:.4f}"
                )
                self.get_logger().debug(
                    f"DEBUG CNODE: Activation for {name}: {activation.activation}"
                )
                node_activations.append(activation.activation)

            self.get_logger().debug(
                f"DEBUG CNODE: Activation list {node_activations}"
            )

            if node_activations:
                prod = float(numpy.prod(node_activations))
            else:
                prod = 0.0

            self.activation.activation = prod
            self.activation.timestamp = self.get_clock().now().to_msg()
            self.get_logger().info(
                f"CNODE_ACTIVATION - {self.name}: Final activation (product)={self.activation.activation:.4f}"
            )
            self.get_logger().debug(
                f"{self.node_type} activation for {self.name} = {self.activation}"
            )
        else:
            # Use existing logic based on a provided activation list
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
