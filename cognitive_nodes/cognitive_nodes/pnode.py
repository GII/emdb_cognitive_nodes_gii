from collections import deque
import rclpy
from rclpy.time import Time
from core.cognitive_node import CognitiveNode
from cognitive_nodes.space import PointBasedSpace
from core.utils import class_from_classname, perception_msg_to_dict, separate_perceptions
from cognitive_node_interfaces.srv import AddPoint, SendSpace, ContainsSpace
from cognitive_node_interfaces.msg import Perception, PerceptionStamped, SuccessRate

class PNode(CognitiveNode):
    """
    PNode class
    """
    def __init__(self, name= 'pnode', class_name = 'cognitive_nodes.pnode.PNode', space_class = None, space = None, history_size=100, **params):
        """
        Constructor for the PNode class.
        
        Initializes a PNode with the given name and registers it in the LTM.
        It also creates a service for adding points to the node.
        
        :param name: The name of the PNode.
        :type name: str
        :param class_name: The name of the PNode class.
        :type class_name: str
        :param space_class: The class of the space used to define the PNode
        :type space_class: str
        :param space: The space used to define the PNode
        :type space: cognitive_nodes.space
        """
        super().__init__(name, class_name, **params)
        self.spaces = [space if space else class_from_classname(
            space_class)(ident=name + " space")]
        self.space=None
        self.added_point = False
        self.add_point_service = self.create_service(AddPoint, 'pnode/' + str(
            name) + '/add_point', self.add_point_callback, callback_group=self.cbgroup_server)
        self.send_pnode_space_service = self.create_service(SendSpace, 'pnode/' + str(
            name) + '/send_space', self.send_pnode_space_callback, callback_group=self.cbgroup_server)
        self.contains_space_service = self.create_service(ContainsSpace, 'pnode/' + str(
            name) + '/contains_space', self.contains_space_callback, callback_group=self.cbgroup_server)
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

        :param request: Empty request
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

        :param request: Request that contains the space to check
        :type request: cognitive_node_interfaces.srv.ContainsSpace.Request
        :param response: Response that indicates if the space is contained
        :type response: cognitive_node_interfaces.srv.ContainsSpace.Response
        :return: Response that indicates if the space is contained
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
        Callback method for adding a point (or anti-point) to a specific PNode.

        :param request: The request that contains the point that is added and its confidence.
        :type request: cognitive_node_interfaces.srv.AddPoint_Request
        :param response: The response indicating if the point was added to the PNode.
        :type respone: core_interfaces.srv.AddPoint_Response
        :return: The response indicating if the point was added to the PNode.
        :rtype: cognitive_node_interfaces.srv.AddPoint_Response
        """
        self.point_msg = request.point
        confidence = request.confidence
        point = perception_msg_to_dict(self.point_msg)
        self.add_point(point,confidence)
        self.get_logger().info('Adding point: ' + str(point) + 'Confidence: ' + str(confidence))
        response.added = True

        return response
    
    def add_point(self, point, confidence):
        """
        Add a new point (or anti-point) to the PNode.
        
        :param point: The point that is added to the PNode
        :type point: dict
        :param confidence: Indicates if the perception added is a point or an antipoint.
        :type confidence: float
        """
        points = separate_perceptions(point)
        for point in points:
            self.space = self.spaces[0]
            if not self.space:
                self.space = self.spaces[0].__class__()
                self.spaces.append(self.space)
            added_point_pos = self.space.add_point(point, confidence)
        self.added_point = True
        self.update_history(confidence)
        self.publish_success_rate()
            
    def calculate_activation(self, perception=None, activation_list=None):
        """
        Calculate the new activation value for a given perception

        :param perception: The perception for which PNode activation is calculated.
        :type perception: dict
        :return: If there is space, returns the activation of the PNode. If not, returns 0
        :rtype: float
        """
        if activation_list!=None:
            perception={}
            for sensor in activation_list:
                activation_list[sensor]['updated']=False
                perception[sensor]=activation_list[sensor]['data']

        if perception:
            activations = []
            perceptions = separate_perceptions(perception)
            for perception_line in perceptions:
                space = self.spaces[0]
                if space and self.added_point:
                    activation_value = max(0.0, space.get_probability(perception_line))
                    self.get_logger().debug(f'PNODE DEBUG: Perception: {perception_line} Space provided activation: {activation_value}')
                else:
                    activation_value = 0.0

                activations.append(activation_value) 
            
            self.activation.activation = activations[0] if len(activations) == 1 else float(max(activations)) #Fix this else case for multiple perceptions
            self.activation.timestamp = self.get_clock().now().to_msg()
        return self.activation

    def get_space(self, perception):
        """
        Return the compatible space with perception.
        (Ugly hack just to see if this works. In that case, everything need to be checked to reduce the number of
        conversions between sensing, perception and space.)

        :param perception: The perception for which PNode activation is calculated.
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

        :param node: Dictionary with the information of the node {'name': <name>, 'node_type': <node_type>}
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
            self.get_logger().warn("Empty perception recieved in PNode. No activation calculated")
    
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

    def process_neighbors(self):
        """
        Detects if the P-Node is linked to a Goal node.
        """        
        goals=[node["name"] for node in self.neighbors if node["node_type"] == "Goal"]
        self.get_logger().debug(f"DEBUG: PNode {self.name} neighbors: {self.neighbors}")
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
    


def main(args = None):
    rclpy.init(args=args)

    pnode = PNode()

    rclpy.spin(pnode)

    pnode.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()