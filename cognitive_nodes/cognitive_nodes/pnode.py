import rclpy
import numpy as np
from collections import deque
from rclpy.time import Time

from core.cognitive_node import CognitiveNode
from cognitive_nodes.space import PointBasedSpace
from core.container import Container, consolidate_containers
from core.utils import class_from_classname

from cognitive_node_interfaces.srv import AddPoints, SendSpace, ContainsSpace, SaveModel
from cognitive_node_interfaces.msg import SuccessRate
from core_interfaces.msg import Container as ContainerMsg

class PNode(CognitiveNode):
    """
    P-Node class
    """
    def __init__(self, name= 'pnode', class_name = 'cognitive_nodes.pnode.PNode', node_type="PNode", space_class = None, space = None, history_size=100, space_parameters=None, **params):
        """
        Constructor for the P-Node class.
        
        Initializes a P-Node with the given name and registers it in the LTM.
        It also creates a service for adding points to the node.
        
        :param name: The name of the P-Node.
        :type name: str
        :param class_name: The name of the P-Node class.
        :type class_name: str
        :param node_type: The type of the node, defaults to "PNode".
        :type node_type: str
        :param space_class: The class of the space used to define the P-Node.
        :type space_class: str
        :param space: The space used to define the P-Node.
        :type space: cognitive_nodes.space
        :param history_size: The size of the history of the P-Node.
        :type history_size: int
        """
        super().__init__(name, class_name, node_type=node_type, **params)
        self.spaces = [space if space else class_from_classname(
            space_class)(ident=name + " space", **(space_parameters if space_parameters else {}))]
        self.space=self.spaces[0]
        self.added_point = False
        self.add_points_service = self.create_service(AddPoints, 'pnode/' + str(
            name) + '/add_points', self.add_points_callback, callback_group=self.cbgroup_server)
        self.send_pnode_space_service = self.create_service(SendSpace, 'pnode/' + str(
            name) + '/send_space', self.send_pnode_space_callback, callback_group=self.cbgroup_server)
        self.contains_space_service = self.create_service(ContainsSpace, 'pnode/' + str(
            name) + '/contains_space', self.contains_space_callback, callback_group=self.cbgroup_server)
        self.save_model_service = self.create_service(SaveModel, "pnode/" + str(
            name) + '/save_model', self.save_model_callback, callback_group=self.cbgroup_server)
        self.perception = None
        self.history_size = history_size
        self.history = deque([], history_size)
        self.success_rate = 0.0
        self.goal_linked = False
        self.success_publisher = self.create_publisher(
            SuccessRate, f'pnode/{str(name)}/success_rate', 0)
        self.configure_activation_inputs(self.neighbors)
        self.data_labels = []

    def send_pnode_space_callback(self, request, response):
        """
        Callback that sends the space of the P-Node.

        :param request: Empty request.
        :type request: cognitive_node_interfaces.srv.SendGoalSpace.Request
        :param response: Response that contains the space of the P-Node.
        :type response: cognitive_node_interfaces.srv.SendGoalSpace.Response
        :return: Response that contains the space of the P-Node.
        :rtype: cognitive_node_interfaces.srv.SendGoalSpace.Response
        """
        if self.space:
            response.space = self.space.to_msg()
        return response
    
    def contains_space_callback(self, request, response):
        """
        Callback that checks if the space contains a given space.

        :param request: Request that contains the space to check.
        :type request: cognitive_node_interfaces.srv.ContainsSpace.Request
        :param response: Response that indicates if the space is contained.
        :type response: cognitive_node_interfaces.srv.ContainsSpace.Response
        :return: Response that indicates if the space is contained.
        :rtype: cognitive_node_interfaces.srv.ContainsSpace.Response
        """
        space_data = Container.from_msg(request.space)
        if self.space:
            response.contained=self.space.contains(space_data)
        else:
            response.contained=False
        return response

    def add_points_callback(self, request, response):
        """
        Callback method for adding a point (or anti-point) to a specific P-Node.

        :param request: The request that contains the list of points that are added and their confidence.
        :type request: cognitive_node_interfaces.srv.AddPoints.Request
        :param response: The response indicating if the points were added to the P-Node.
        :type respone: core_interfaces.srv.AddPoints.Response
        :return: The response indicating if the points were added to the P-Node.
        :rtype: cognitive_node_interfaces.srv.AddPoints.Response
        """
        if request.points:
            points = Container.from_msg(request.points) 
            confidences = np.asarray(request.confidences)
            if len(points) != len(confidences):
                self.get_logger().error(f"Number of points and confidences do not match. Points: {len(points)}, Confidences: {len(confidences)}")
                response.added = False
                return response
            self.add_points(points, confidences)
            response.added = True
            self.get_logger().info(f'Added: {len(points)} points with mean confidence: {np.mean(confidences)}')
        else:
            response.added = False
        return response
    
    def add_points(self, points, confidences):
        """
        Add new points (or anti-points) to the P-Node.

        :param points: The points that are added to the P-Node.
        :type points: core.container.Container
        :param confidences: The confidences associated with each point.
        :type confidences: numpy.ndarray
        """
        if not self.space:
            self.get_logger().error("No space defined for the P-Node. Cannot add points.")
            return
        self.space.add_point(points, confidences)
        self.added_point = True
        for confidence in confidences:
            self.update_history(confidence)
        self.publish_success_rate()
        self.get_logger().info(f"P-Node success rate: {self.success_rate}")
        return True
            
    def calculate_activation(self, perception=None, activation_list=None):
        """
        Calculate the new activation value for a given perception.

        :param perception: The perception for which P-Node activation is calculated.
        :type perception: container.Container
        :param activation_list: The list of activations to be used for the calculation.
        :type activation_list: list
        :return: If there is space, returns the activation of the P-Node. If not, returns 0. 
            It also returs the timestamp.
        :rtype: cognitive_node_interfaces.msg.Activation
        """
        confidences = None
        if activation_list!=None:
            # Uses internal readings to calculate the activation. This is used in normal execution of the architecture.
            data = [activation_list[sensor]['data'] for sensor in activation_list]
            if self.perception is None and len(data)>0:
                self.perception = consolidate_containers(data, name="perception", container_type="perception")
            elif len(data)==0: # Activation list may be empty when initializing the P-Node.
                self.activation.activation = 0.0
                self.activation.timestamp = self.get_clock().now().to_msg()
                return self.activation
            else:
                consolidate_containers(data, write_container=self.perception)
            space_activation = self.space.get_probability(self.perception) if self.space else np.zeros(len(self.perception))
            activation_value = max(0.0, space_activation.reshape(-1)[0])
            self.activation.activation = float(activation_value)
            perception_timestamp = self.perception.data.coords["timestamp"].values[-1]
            self.activation.timestamp = Time(nanoseconds=perception_timestamp).to_msg()
            return self.activation
        
        if perception:
            # Uses the provided perception to calculate the activation. This is used when the activation is requested from outside (e.g., get_activation service).
            activations = self.space.get_probability(perception).reshape(-1) if self.space else np.zeros(len(perception))
            return activations.tolist()

    def calculate_confidence(self, perception=None, activation_list=None):
        """
        This method use the already calculated success rate as confidence value, 
        stored in 
        """
        return self.success_rate
    
    def create_activation_input(self, node: dict): #Adds or deletes a node from the activation inputs list. By default reads activations.
        """
        Adds perceptions to the activation inputs list.

        :param node: Dictionary with the information of the node {'name': <name>, 'node_type': <node_type>}.
        :type node: dict
        """    
        name=node['name']
        node_type=node['node_type']
        if node_type == "Perception":
            subscriber=self.create_subscription(ContainerMsg, "perception/" + str(name) + "/value", self.read_activation_callback, 1, callback_group=self.cbgroup_activation)
            data=None
            updated=False
            new_input=dict(subscriber=subscriber, data=data, updated=updated)
            self.activation_inputs[name]=new_input
            self.get_logger().debug(f'{self.name} -- Created new activation input: {name} of type {node_type}')

    def read_activation_callback(self, msg: ContainerMsg):
        """
        Callback method that reads a perception and stores it in the activation inputs list.

        :param msg: PerceptionStamped message that contains the perception and its timestamp.
        :type msg: cognitive_node_interfaces.msg.PerceptionStamped
        """        
        if msg.max_size>1:
            self.get_logger().error(f'Received perception with multiple readings: ({msg.name}). Perception messages should (currently) include only one reading!')
        elif msg.max_size==1:
            node_name=msg.name
            if node_name in self.activation_inputs:
                if self.activation_inputs[node_name]['data'] is None:
                    self.activation_inputs[node_name]['data']=Container.from_msg(msg)
                else:
                    self.activation_inputs[node_name]['data'].push_from_msg(msg)
                self.activation_inputs[node_name]['updated']=True
            else:
                self.get_logger().error(
                    "Received perception not registered in local perception cache!!!"
                )
        else:
            self.get_logger().warn("Empty perception recieved in P-Node")
    
    def add_neighbor_callback(self, request, response):
        """
        Extends the default add_neighbor_callback method to process the neighbors and publish the success rate.

        :param request: Add neighbor request.
        :type request: cognitive_node_interfaces.srv.AddNeighbor.Request
        :param response: Response with the result of the add neighbor operation.
        :type response: cognitive_node_interfaces.srv.AddNeighbor.Response
        :return: Response with the result of the add neighbor operation.
        :rtype: cognitive_node_interfaces.srv.AddNeighbor.Response
        """        
        response = super().add_neighbor_callback(request, response)
        self.process_neighbors()
        self.publish_success_rate()
        return response
    
    def delete_neighbor_callback(self, request, response):
        """
        Extends the default delete_neighbor_callback method to process the neighbors and publish the success rate.

        :param request: Delete neighbor request.
        :type request: cognitive_node_interfaces.srv.DeleteNeighbor.Request
        :param response: Response with the result of the delete neighbor operation.
        :type response: cognitive_node_interfaces.srv.DeleteNeighbor.Response
        :return: Response with the result of the delete neighbor operation.
        :rtype: cognitive_node_interfaces.srv.DeleteNeighbor.Response
        """       
        response = super().delete_neighbor_callback(request, response)
        self.process_neighbors()
        self.publish_success_rate()
        return response

    def save_model_callback(self, request, response):
        """
        Save the current model to a file.

        :param request: The request that contains the prefix and suffix for the file name.
        :type request: cognitive_node_interfaces.srv.SaveModel.Request
        :param response: The response that contains the saved model path and success status.
        :type response: cognitive_node_interfaces.srv.SaveModel.Response
        :return: The response that contains the saved model path and success status.
        :rtype: cognitive_node_interfaces.srv.SaveModel.Response
        """
        self.get_logger().info('Saving model...')
        if self.space is not None and hasattr(self.space, 'save_model'):
            model_name = f"{request.prefix}{self.name}{request.suffix}"
            try:
                success, path = self.space.save_model(model_name)
            except Exception as e:
                self.get_logger().error(f"Error saving model: {e}")
                path = ""
                success = False
            response.saved_model_path = path
            response.success = success
            if success:
                self.get_logger().info(f"Model saved to {path}.")
            else:
                self.get_logger().error("Failed to save model.")
        else:
            response.saved_model_path = ""
            response.success = False
            self.get_logger().error("Learner does not support saving models.")
        return response


    def process_neighbors(self):
        """
        Detects if the P-Node is linked to a Goal node.
        """        
        goals=[node["name"] for node in self.neighbors if node["node_type"] == "Goal"]
        self.get_logger().debug(f"DEBUG: P-Node {self.name} neighbors: {self.neighbors}")
        if len(goals)>0:
            self.goal_linked=True
        else:
            self.goal_linked=False

    def publish_success_rate(self):
        """
        Publishes the success rate of the P-Node.
        """        
        msg = SuccessRate()
        msg.node_name=self.name
        msg.node_type=self.node_type
        msg.flag=self.goal_linked
        msg.success_rate=self.success_rate
        self.success_publisher.publish(msg)

    def update_history(self, confidence):
        """
        Updates the history of the P-Node with the new confidence value (point or anti-point).

        :param confidence: Confidence value of the new point or anti-point.
        :type confidence: int
        """        
        if confidence>0 and self.space.learnable():
            self.history.appendleft(True)
        else:
            self.history.appendleft(False)
        self.success_rate = sum(self.history)/self.history.maxlen
        self.get_logger().debug(f"DEBUG: Added point with confidence: {confidence}. New success rate: {self.success_rate}. Learnable: {self.space.learnable()}")


def main(args = None):
    rclpy.init(args=args)

    pnode = PNode()

    rclpy.spin(pnode)

    pnode.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()