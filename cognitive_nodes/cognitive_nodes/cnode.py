import numpy as np
import rclpy
from collections import deque

from core.cognitive_node import CognitiveNode
from core.service_client import ServiceClientAsync
from cognitive_node_interfaces.srv import GetActivation, LogExecution


class CNode(CognitiveNode):
    """
    C-Node class.
    It represents a context, i.e. a link between nodes that were activated together in the past.
    It is assumed that there is only one element of each type connected to the C-Node.
    """

    def __init__(self, name="cnode", class_name="cognitive_nodes.cnode.CNode", node_type="CNode", history_size=40, track_competence=True, default_competence=1.0, **params):
        """
        Constructor of the C-Node class.
        Initializes a C-Node with the given name and registers it in the LTM.

        :param name: The name of the C-Node.
        :type name: str
        :param class_name: The name of the C-Node class.
        :type class_name: str
        :param node_type: The type of the C-Node.
        :type node_type: str
        :param history_size: The size of the history of the C-Node.
        :type history_size: int
        """
        super().__init__(name, class_name, node_type=node_type, **params)
        self.configure_activation_inputs(self.neighbors)
        self.history = deque(np.zeros(history_size), maxlen=history_size)
        self.track_competence = track_competence
        self.default_competence = default_competence
        self.calculate_metacognitive_parameters()
        self.metacognitive_params["confidence"] = 1.0

        self.execution_log_service = self.create_service(
            LogExecution,
            "cnode/" + str(self.name) + "/log_execution",
            self.log_execution_callback,
            callback_group=self.cbgroup_server,
        )

    def log_execution_callback(self, request, response):
        """
        Callback to log the execution of a C-Node.
        It receives a request with the execution data and logs it in the C-Node's history.

        :param request: LogExecution request message.
        :type request: cognitive_node_interfaces.srv.LogExecution.Request
        :param response: LogExecution response message.
        :type response: cognitive_node_interfaces.srv.LogExecution.Response
        :return: The response message.
        :rtype: cognitive_node_interfaces.srv.LogExecution.Response
        """
        self.history.append(request.success)
        self.calculate_metacognitive_parameters()
        return response

    def calculate_metacognitive_parameters(self):
        """
        Calculate the metacognitive parameters of the C-Node.
        It calculates the competence and delta competence of the C-Node based on its history.
        """
        if not self.track_competence:
            self.metacognitive_params["competence"] = self.default_competence
            self.metacognitive_params["delta_competence"] = 0.0
        else:
            self.get_logger().info(
                f"Calculating metacognitive parameters for {self.node_type} {self.name}..."
            )

            history = np.asarray(self.history, dtype=float)
            sample_count = history.size


            midpoint = sample_count // 2
            old_history = history[:midpoint]
            new_history = history[midpoint:]

            old_competence = np.mean(old_history) if old_history.size > 0 else 0.0
            new_competence = np.mean(new_history) if new_history.size > 0 else 0.0
            overall_competence = np.mean(history)
            delta_competence = (new_competence - old_competence)
            self.metacognitive_params["competence"] = overall_competence
            self.metacognitive_params["delta_competence"] = delta_competence

            self.get_logger().info(
                f"Competence: {overall_competence}, Delta Competence: {delta_competence} "
                f"for {self.node_type} {self.name}."
            )

    async def calculate_activation(self, perception=None, activation_list=None):
        """
        Calculate the new activation value.

        If activation_list is None:
          - Request the activation of all neighbors (except Policies) for the given perception.
          - Compute the product of their activations.
        If activation_list is provided:
          - Delegate to calculate_activation_prod to use the cached activations.

        :param perception: Arbitrary perception to propagate to neighbors.
        :type perception: core.container.Container
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
            perception_msg = perception.to_msg()
            for name in neighbors_name:
                service_name = "cognitive_node/" + str(name) + "/get_activation"
                if not service_name in self.node_clients:
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
            self.get_logger().debug(f"DEBUG CNODE: Activation list {node_activations}")
            activation_list = np.prod(node_activations)
            self.activation.activation = float(np.max(activation_list))
            self.activation.timestamp=self.get_clock().now().to_msg()
            activation = await self.node_clients[service_name].send_request_async(
                perception=perception_msg
            )
            self.get_logger().debug(f"DEBUG CNODE: Activation for {name}: {activation.activation}.")
            node_activations.append(activation.activation)
            self.get_logger().debug(f"DEBUG CNODE: Activation list {node_activations}.")
            activation_list = np.prod(node_activations)
            self.activation = np.max(activation_list)

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
