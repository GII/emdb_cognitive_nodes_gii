import rclpy
import numpy as np
from rclpy.time import Time
from collections import deque

from core.cognitive_node import CognitiveNode
from cognitive_nodes.space import PointBasedSpace
from core.utils import class_from_classname, perception_msg_to_dict, separate_perceptions
from cognitive_node_interfaces.srv import AddPoint, AddPoints, SendSpace, ContainsSpace, SaveModel
from cognitive_node_interfaces.msg import Perception, PerceptionStamped, SuccessRate

class PNode(CognitiveNode):
    """
    P-Node class
    """
    def __init__(self, name= 'pnode', class_name = 'cognitive_nodes.pnode.PNode', space_class = None, space = None, history_size=100, **params):
        """
        Constructor for the P-Node class.
        
        Initializes a P-Node with the given name and registers it in the LTM.
        It also creates a service for adding points to the node.
        
        :param name: The name of the P-Node.
        :type name: str
        :param class_name: The name of the P-Node class.
        :type class_name: str
        :param space_class: The class of the space used to define the P-Node.
        :type space_class: str
        :param space: The space used to define the P-Node.
        :type space: cognitive_nodes.space
        :param history_size: The size of the history of the P-Node.
        :type history_size: int
        """
        super().__init__(name, class_name, **params)
        self.spaces = [space if space else class_from_classname(
            space_class)(ident=name + " space")]
        self.space=None
        self.added_point = False
        self.add_point_service = self.create_service(AddPoint, 'pnode/' + str(
            name) + '/add_point', self.add_point_callback, callback_group=self.cbgroup_server)
        self.add_points_service = self.create_service(AddPoints, 'pnode/' + str(
            name) + '/add_points', self.add_points_callback, callback_group=self.cbgroup_server)
        self.send_pnode_space_service = self.create_service(SendSpace, 'pnode/' + str(
            name) + '/send_space', self.send_pnode_space_callback, callback_group=self.cbgroup_server)
        self.contains_space_service = self.create_service(ContainsSpace, 'pnode/' + str(
            name) + '/contains_space', self.contains_space_callback, callback_group=self.cbgroup_server)
        self.save_model_service = self.create_service(SaveModel, "pnode/" + str(
            name) + '/save_model', self.save_model_callback, callback_group=self.cbgroup_server)
        self.history_size = history_size
        self.history = deque([], history_size)
        self.success_rate = 0.0
        self.goal_linked = False
        self.success_publisher = self.create_publisher(
            SuccessRate, f'pnode/{str(name)}/success_rate', 0)
        self.configure_activation_inputs(self.neighbors)
        self.data_labels = []

    def configure_labels(self): #TODO This method creates one label for each sensor even if there are multiple objects in the sensor. Spaces use separated perceptions. 
        """
        Configure the labels of the space.
        """  
        self.point_msg:Perception
        i = 0
        for dim in self.point_msg.layout.dim:
            sensor = dim.object[:-1]
            for label in dim.labels:
                data_label = str(i) + "-" + sensor + "-" + label
                self.data_labels.append(data_label)
            i = i+1            

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
            if not self.data_labels:
                self.configure_labels()
            response.labels = self.data_labels
            
            data = []
            for perception in self.space.members[0:self.space.size]:
                for value in perception:
                    data.append(value)
            response.data = data

            confidences = list(self.space.memberships[0:self.space.size])
            response.confidences = confidences
            
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
        labels=request.labels
        data = request.data  # Flattened list of data values
        confidences = request.confidences  # List of confidence values
        compare_space=PointBasedSpace(len(confidences))
        compare_space.populate_space(labels, data, confidences)
        if self.space:
            response.contained=self.space.contains(compare_space)
        else:
            response.contained=False
        return response   

    def add_point_callback(self, request, response):
        """
        DEPRECATED: SEE add_points_callback
        Callback method for adding a point (or anti-point) to a specific P-Node.

        :param request: The request that contains the point that is added and its confidence.
        :type request: cognitive_node_interfaces.srv.AddPoint.Request
        :param response: The response indicating if the point was added to the P-Node.
        :type response: cognitive_node_interfaces.srv.AddPoint.Response
        :return: The response indicating if the point was added to the P-Node.
        :rtype: cognitive_node_interfaces.srv.AddPoint.Response
        """
        self.point_msg = request.point
        confidence = request.confidence
        point = perception_msg_to_dict(self.point_msg)
        response.added = self.add_point(point,confidence)
        if response.added:
            self.get_logger().info('Adding point: ' + str(point) + 'Confidence: ' + str(confidence))
        else:
            self.get_logger().warn(f'Ignored empty/invalid point for {self.name}: {point}')

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
            self.point_msg = request.points[0]
            added_count = 0
            for point, confidence in zip(request.points, request.confidences):
                point_dict = perception_msg_to_dict(point)
                if self.add_point(point_dict, confidence):
                    added_count += 1
            response.added = added_count > 0
            self.get_logger().info(f'Added: {added_count}/{len(request.points)} points with mean confidence: {np.mean(request.confidences)}')
        else:
            response.added = False
        return response
    
    def add_point(self, point, confidence):
        """
        Add a new point (or anti-point) to the P-Node.
        
        :param point: The point that is added to the P-Node.
        :type point: dict
        :param confidence: Indicates if the perception added is a point or an antipoint.
        :type confidence: float
        """
        points = separate_perceptions(point)
        if not points:
            return False

        for point in points:
            self.space = self.spaces[0]
            if not self.space:
                self.space = self.spaces[0].__class__()
                self.spaces.append(self.space)
            self.space.add_point(point, confidence)
        self.added_point = True
        self.update_history(confidence)
        self.publish_success_rate()
        self.get_logger().info(f"P-Node success rate: {self.success_rate}")
        return True
            
    def calculate_activation(self, perception=None, activation_list=None):
        """
        Calculate the new activation value for a given perception.

        :param perception: The perception for which P-Node activation is calculated.
        :type perception: dict
        :param activation_list: The list of activations to be used for the calculation.
        :type activation_list: list
        :return: If there is space, returns the activation of the P-Node. If not, returns 0. 
            It also returs the timestamp.
        :rtype: cognitive_node_interfaces.msg.Activation
        """
        confidences = None
        if activation_list!=None:
            perception={}
            confidences = []
            for sensor in activation_list:
                activation_list[sensor]['updated']=False
                perception[sensor]=activation_list[sensor]['data']
                # Use provided confidence if present, otherwise default to 1.0
                confidences.append(float(activation_list[sensor].get('confidence', 1.0)))

        if perception:
            activations = []
            perceptions = separate_perceptions(perception)
            if not perceptions:
                self.activation.activation = 0.0
                self.activation.timestamp = self.get_clock().now().to_msg()
                return self.activation

            for idx, perception_line in enumerate(perceptions):
                space = self.spaces[0]
                if space and self.added_point:
                    base_activation = max(0.0, space.get_probability(perception_line))
                    # weight activation by confidence (if provided), default 1.0
                    conf = confidences[idx] if confidences is not None and idx < len(confidences) else 1.0
                    activation_value = float(base_activation) * float(conf)
                    if activation_list is None:
                        self.get_logger().info(f'PNODE DEBUG: Perception: {perception_line} Space provided activation: {activation_value}')
                else:
                    activation_value = 0.0

                activations.append(activation_value)

            # If multiple perceptions, take the max weighted activation
            self.activation.activation = activations[0] if len(activations) == 1 else float(max(activations))
            self.activation.timestamp = self.get_clock().now().to_msg()
        return self.activation
    
    def calculate_confidence(self, perception=None, activation_list=None):
        """
        This method use the already calculated success rate as confidence value, 
        stored in 
        """
        return self.success_rate

    def get_space(self, perception):
        """
        Return the compatible space with perception.
        (Ugly hack just to see if this works. In that case, everything need to be checked to reduce the number of
        conversions between sensing, perception and space).

        :param perception: The perception for which P-Node activation is calculated.
        :type perception: dict
        :return: If there is space, returns it. If not, returns None.
        :rtype: cognitive_nodes.space or None
        """
        temp_space = self.spaces[0].__class__()
        temp_space.add_point(perception, 1.0)
        for space in self.spaces:
            if (not space.size) or space.same_sensors(temp_space):
                return space
        return None
    
    def create_activation_input(self, node: dict): #Adds or deletes a node from the activation inputs list. By default reads activations.
        """
        Adds perceptions to the activation inputs list.

        :param node: Dictionary with the information of the node {'name': <name>, 'node_type': <node_type>}.
        :type node: dict
        """    
        name=node['name']
        node_type=node['node_type']
        if node_type == "Perception":
            subscriber=self.create_subscription(PerceptionStamped, "perception/" + str(name) + "/value", self.read_activation_callback, 1, callback_group=self.cbgroup_activation)
            data=Perception()
            updated=False
            timestamp=Time()
            new_input=dict(subscriber=subscriber, data=data, updated=updated, timestamp=timestamp)
            self.activation_inputs[name]=new_input
            self.get_logger().debug(f'{self.name} -- Created new activation input: {name} of type {node_type}')


    def read_activation_callback(self, msg: PerceptionStamped):
        """
        Callback method that reads a perception and stores it in the activation inputs list.

        :param msg: PerceptionStamped message that contains the perception and its timestamp.
        :type msg: cognitive_node_interfaces.msg.PerceptionStamped
        """        
        perception_dict=perception_msg_to_dict(msg=msg.perception)
        if len(perception_dict)>1:
            self.get_logger().error(f'{self.name} -- Received perception with multiple sensors: ({perception_dict.keys()}). Perception nodes should (currently) include only one sensor!')
        if len(perception_dict)==1:
            node_name=list(perception_dict.keys())[0]
            if node_name in self.activation_inputs:
                self.activation_inputs[node_name]['data']=perception_dict[node_name]
                self.activation_inputs[node_name]['updated']=True
                self.activation_inputs[node_name]['timestamp']=Time.from_msg(msg.timestamp)
        else:
            self.get_logger().warn("Empty perception recieved in P-Node. No activation calculated")
    
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
        self.get_logger().info(f"DEBUG: Added point with confidence: {confidence}. New success rate: {self.success_rate}. Learnable: {self.space.learnable()}")


def main(args = None):
    rclpy.init(args=args)

    pnode = PNode()

    rclpy.spin(pnode)

    pnode.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()