import rclpy
import numpy as np
import tempfile
from pathlib import Path
from collections import deque, OrderedDict
from rclpy.time import Time

from core.service_client import ServiceClientAsync
from core.cognitive_node import CognitiveNode
from cognitive_nodes.space import PointBasedSpace
from core.container import Container, consolidate_containers
from core.utils import class_from_classname

from cognitive_node_interfaces.srv import AddPoints, SendSpace, ContainsSpace, SaveModel, LoadSpace
from cognitive_node_interfaces.msg import SuccessRate
from core_interfaces.msg import Container as ContainerMsg

ACTIVATION_CACHE_SIZE = 16  # Perceptions whose activations each P-Node keeps (see cached_probability)

class PNode(CognitiveNode):
    """
    P-Node class.

    A P-Node (perceptual node) represents a region of the perceptual space through a space object
    (:class:`cognitive_nodes.space.PointBasedSpace` or a subclass). Its activation is the activation of the space for the
    current perception, clipped to be at least 0. Points and antipoints are added to the space through the
    ``pnode/<name>/add_points`` service or :meth:`add_points`.

    The node keeps a history of the last ``history_size`` points added, in which a point counts as a success if its
    confidence is positive and the space is configured. The success rate (fraction of successes over ``history_size``) is
    published in ``pnode/<name>/success_rate`` and is the node's ``confidence`` metacognitive parameter.

    The activations of the space are cached in ``activation_cache``, an ``OrderedDict`` of at most ``ACTIVATION_CACHE_SIZE``
    (16) entries. The key is made of the feature labels, the shape and the values of the perception, without the timestamps,
    because the timestamps change with every reading even if the values do not. The oldest entry is discarded
    when the cache is full (first in, first out: reading an entry does not renew it). The cache is emptied by
    :meth:`space_changed`, which is called when points are added or a space is loaded. See :meth:`cached_probability`.

    The node reads its perceptions from the activation inputs (subscriptions to ``perception/<name>/value``), which are
    consolidated into a single container before each calculation.
    """
    def __init__(self, name= 'pnode', class_name = 'cognitive_nodes.pnode.PNode', node_type="PNode", space_class = None, space = None, history_size=100, space_parameters=None, **params):
        """
        Constructor for the P-Node class.
        
        Initializes a P-Node with the given name and registers it in the LTM.
        It also creates the services for adding points, loading and sending the space, checking if the space contains
        another one and saving the model of the space, all under ``pnode/<name>/``, and the publisher of the success rate.
        
        :param name: The name of the P-Node.
        :type name: str
        :param class_name: The name of the P-Node class.
        :type class_name: str
        :param node_type: The type of the node, defaults to "PNode".
        :type node_type: str
        :param space_class: Full name of the class of the space used to define the P-Node (e.g. ``cognitive_nodes.space.SVMSpace``). It is only used if no space is given.
        :type space_class: str
        :param space: The space used to define the P-Node. If given, it is used instead of creating one from space_class.
        :type space: cognitive_nodes.space.PointBasedSpace
        :param history_size: The size of the history of the P-Node, which is also the denominator of the success rate. Defaults to 100.
        :type history_size: int
        :param space_parameters: Keyword arguments for the constructor of the space class, in addition to ``ident`` (``"<name> space"``). If it does not have ``random_seed``, the random seed of the node is used (0 if it has none). They are registered so that duplicates of the node are created with them.
        :type space_parameters: dict
        :param params: Additional parameters for :class:`core.cognitive_node.CognitiveNode`.
        :type params: dict
        """
        super().__init__(name, class_name, node_type=node_type, **params)
        # Forward the node's random_seed to the space so a configured seed makes
        # the space reproducible too (an explicit random_seed in space_parameters
        # takes precedence).
        space_parameters = dict(space_parameters) if space_parameters else {}
        space_parameters.setdefault('random_seed', getattr(self, 'random_seed', 0))
        self.spaces = [space if space else class_from_classname(
            space_class)(ident=name + " space", **space_parameters)]
        self.space=self.spaces[0]
        self.activation_cache = OrderedDict()  # Activations of the space by perception values (see cached_probability)
        self.added_point = False
        self.add_points_service = self.create_service(AddPoints, 'pnode/' + str(
            name) + '/add_points', self.add_points_callback, callback_group=self.cbgroup_server)
        self.load_points_service = self.create_service(LoadSpace, 'pnode/' + str(
            name) + '/load_space', self.load_space_callback, callback_group=self.cbgroup_server)
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
        self.register_duplicate_parameters(
            space_class=space_class,
            history_size=history_size,
            space_parameters=space_parameters,
        )

    def send_pnode_space_callback(self, request, response):
        """
        Callback that sends the space of the P-Node.

        :param request: Empty request.
        :type request: cognitive_node_interfaces.srv.SendSpace.Request
        :param response: Response that contains the space of the P-Node.
        :type response: cognitive_node_interfaces.srv.SendSpace.Response
        :return: Response that contains the space of the P-Node.
        :rtype: cognitive_node_interfaces.srv.SendSpace.Response
        """
        if self.space:
            response.space = self.space.to_msg()
        return response
    
    def contains_space_callback(self, request, response):
        """
        Callback that checks if the space contains a given space.

        ``contained`` is False if the node has no space or the received data is None; otherwise it is the result of ``contains``
        of the space (see :meth:`cognitive_nodes.space.PointBasedSpace.contains`).

        :param request: Request that contains the space to check.
        :type request: cognitive_node_interfaces.srv.ContainsSpace.Request
        :param response: Response that indicates if the space is contained.
        :type response: cognitive_node_interfaces.srv.ContainsSpace.Response
        :return: Response that indicates if the space is contained.
        :rtype: cognitive_node_interfaces.srv.ContainsSpace.Response
        """
        space_data = Container.from_msg(request.space)
        if self.space and space_data is not None:
            response.contained=self.space.contains(space_data)
        else:
            response.contained=False
        return response

    def add_points_callback(self, request, response):
        """
        Callback method for adding a point (or anti-point) to a specific P-Node.

        The points are rejected (``added`` is False) if the request has none or if their number differs from the number of confidences.

        :param request: The request that contains the list of points that are added and their confidence.
        :type request: cognitive_node_interfaces.srv.AddPoints.Request
        :param response: The response indicating if the points were added to the P-Node.
        :type response: cognitive_node_interfaces.srv.AddPoints.Response
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

    def load_space_callback(self, request, response):
        """
        Callback method for loading a complete space into a specific P-Node.

        If the space of the P-Node has no data yet, the loaded container becomes its data. Otherwise the loaded feature labels
        must be equal to those of the P-Node's space (else nothing is loaded and ``loaded`` is False); the data is then
        cleared and replaced by the loaded points. In both cases, the activation cache is emptied.

        :param request: The request that contains the space to be loaded.
        :type request: cognitive_node_interfaces.srv.LoadSpace.Request
        :param response: The response indicating if the space was loaded into the P-Node.
        :type response: cognitive_node_interfaces.srv.LoadSpace.Response
        :return: The response indicating if the space was loaded into the P-Node.
        :rtype: cognitive_node_interfaces.srv.LoadSpace.Response
        """
        space_data = Container.from_msg(request.space)
        if space_data is None:
            self.get_logger().error(f"Failed to load space into P-Node {self.name}. Space data provided is empty or invalid.")
            response.loaded=False
            return response
        if self.space and self.space._data is None:
            space_data.name = self.space.ident + "_data"
            self.space._data = space_data
            self.space_changed()
            self.get_logger().info(f"Loaded space into P-Node {self.name}.")
            response.loaded=True
        elif self.space and self.space._data is not None:
            space_data.name = self.space.ident + "_data"
            if self.space._data.feature_labels != space_data.feature_labels:
                self.get_logger().error(f"Feature labels of the loaded space do not match the P-Node's space. P-Node feature labels: {self.space._data.feature_labels}, Loaded space feature labels: {space_data.feature_labels}")
                response.loaded=False
                return response
            self.space._data.clear()
            self.space._data.push(space_data)
            self.space_changed()
            self.get_logger().info(f"Loaded space into P-Node {self.name}.")
            response.loaded=True
        else:
            self.get_logger().error(f"Failed to load space into P-Node {self.name}.")
            response.loaded=False
        return response
    
    def add_points(self, points, confidences):
        """
        Add new points (or anti-points) to the P-Node.

        The points are passed to ``add_point`` of the space (which may discard antipoints, and may train its model), the
        activation cache is emptied, the confidence of every received point is added to the history (including
        those that the space discarded) and the success rate is published.

        :param points: The points that are added to the P-Node.
        :type points: core.container.Container
        :param confidences: The confidences associated with each point.
        :type confidences: numpy.ndarray
        :return: True after adding the points; None (with an error log) if the node has no space.
        :rtype: bool | None
        """
        if not self.space:
            self.get_logger().error("No space defined for the P-Node. Cannot add points.")
            return
        self.space.add_point(points, confidences)
        self.space_changed()
        self.added_point = True
        for confidence in confidences:
            self.update_history(confidence)
        self.publish_success_rate()
        self.get_logger().info(f"P-Node success rate: {self.success_rate}")
        return True
            
    def calculate_activation(self, perception=None, activation_list=None):
        """
        Calculate the new activation value for a given perception.

        There are three cases:

        - ``activation_list`` is given with perceptions: the data of every input is consolidated into ``self.perception`` (created the first time and rewritten in place afterwards, until an input is added or deleted), and the activations of the space for it are obtained with :meth:`cached_probability`. The activation of the node is the first value, or 0.0 if it is negative, and its timestamp comes from the last timestamp of the perception. The node's ``Activation`` message is returned. If the node has no space, the activation is 0.
        - ``activation_list`` is given but contains no data (it may be empty while the node is initializing): the activation is 0.0 with the current time, and the ``Activation`` message is returned.
        - ``activation_list`` is None and ``perception`` is given: the activations of the space for all the rows of that perception are returned as a list, without clipping them (or zeros if there is no space). This is used when the activation is requested from outside.

        If neither is given, None is returned.

        :param perception: The perception for which P-Node activation is calculated.
        :type perception: core.container.Container
        :param activation_list: Dictionary of the activation inputs, with the data of each input (``activation_list[name]['data']``).
        :type activation_list: dict
        :return: The ``Activation`` message of the node in the first two cases; a list of floats in the third; None if neither argument is given.
        :rtype: cognitive_node_interfaces.msg.Activation | list[float] | None
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
            space_activation = self.cached_probability(self.perception) if self.space else np.zeros(len(self.perception))
            activation_value = max(0.0, space_activation.reshape(-1)[0])
            self.activation.activation = float(activation_value)
            perception_timestamp = self.perception.data.coords["timestamp"].values[-1]
            self.activation.timestamp = Time(nanoseconds=perception_timestamp).to_msg()
            return self.activation
        
        if perception:
            # Uses the provided perception to calculate the activation. This is used when the activation is requested from outside (e.g., get_activation service).
            activations = self.cached_probability(perception).reshape(-1) if self.space else np.zeros(len(perception))
            return activations.tolist()

    def cached_probability(self, perception):
        """
        Activations of the space for the rows of a perception. They only depend on the values of the
        perception and on the space, so they are calculated once and reused while neither changes (e.g.
        the perception, which is published every few milliseconds, between two policy executions, or the
        same perception asked by several goals). The timestamps of the perception are not part of the key:
        they change with every reading, even when the values don't.

        Used by :meth:`calculate_activation`. The key is the tuple of the feature labels, the shape of the values and their
        bytes. On a miss, the activations are calculated with ``get_probability`` of the space and stored; if the cache then
        has more than ``ACTIVATION_CACHE_SIZE`` entries, the oldest one is discarded. The array returned on a hit is the one
        stored in the cache, so it must not be modified. The node must have a space.

        :param perception: The perception.
        :type perception: core.container.Container
        :return: The activation of each row of the perception.
        :rtype: numpy.ndarray
        """
        values, _ = perception.read_numpy()
        key = (tuple(perception.feature_labels), values.shape, values.tobytes())
        activations = self.activation_cache.get(key)
        if activations is None:
            activations = self.space.get_probability(perception)
            self.activation_cache[key] = activations
            if len(self.activation_cache) > ACTIVATION_CACHE_SIZE:
                self.activation_cache.popitem(last=False)
        return activations

    def space_changed(self):
        """
        Discards the cached activations. Called whenever the space of the P-Node changes (points added,
        which may also train its model, or space loaded), in :meth:`add_points` and :meth:`load_space_callback`.
        """
        self.activation_cache.clear()

    def calculate_metacognitive_params(self):
        """
        Calculate the metacognitive parameters of the P-Node and store them in metacognitive_params: its
        confidence is its success rate (the last value calculated by :meth:`update_history`).
        """
        self.metacognitive_params["confidence"] = self.success_rate
    
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
            # Rebuild the perception container with the new set of perceptions in the next activation.
            self.perception = None
            self.get_logger().debug(f'{self.name} -- Created new activation input: {name} of type {node_type}')

    def delete_activation_input(self, node: dict):
        """
        Deletes a perception from the activation inputs list.

        :param node: Dictionary with the information of the node {'name': <name>, 'node_type': <node_type>}.
        :type node: dict
        """
        super().delete_activation_input(node)
        # Rebuild the perception container with the new set of perceptions in the next activation.
        self.perception = None

    def read_activation_callback(self, msg: ContainerMsg):
        """
        Callback method that reads a perception and stores it in the activation inputs list.

        :param msg: Container message that contains the perception and its timestamps.
        :type msg: core_interfaces.msg.Container
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

        The entry is True if ``confidence`` is greater than 0 and the space is configured, and False otherwise. The success
        rate is the number of True entries divided by ``history_size`` (also while the history is not full), and the
        metacognitive parameters are updated.

        :param confidence: Confidence value of the new point or anti-point (a number such as 1.0, 0.0 or -1.0; only its sign matters).
        :type confidence: float
        """        
        if confidence>0 and self.space.configured:
            self.history.appendleft(True)
        else:
            self.history.appendleft(False)
        self.success_rate = sum(self.history)/self.history.maxlen
        self.calculate_metacognitive_params()
        self.get_logger().debug(f"DEBUG: Added point with confidence: {confidence}. New success rate: {self.success_rate}. Configured: {self.space.configured}")

    async def duplicate_node(
        self, name=None, parameters=None, include_neighbors=True
    ):
        """
        Create a duplicate of this node through the commander. P-Nodes include the space model in the duplication process.

        This is a coroutine: it has to be awaited.

        If the space can save a model (``save_model``, as the ANN space), the model is saved to a temporary ``.pth`` file and its path is
        passed to the duplicate as ``model_file`` in its ``space_parameters``, which also merge those of this node with the
        ones in ``parameters`` (the latter prevail). Once the duplicate has been created by
        :meth:`core.cognitive_node.CognitiveNode.duplicate_node`, if the space has points they are sent to the duplicate with
        its ``pnode/<duplicate name>/load_space`` service. The temporary file is deleted at the end.

        :param name: Optional name for the duplicate. If omitted, a unique
            ``<name>_dup_<count>`` name is generated.
        :type name: str
        :param parameters: Optional constructor parameters that override the
            node's duplicate defaults.
        :type parameters: dict
        :param include_neighbors: Include the neighbors of this node in the duplicate, defaults to True.
        :type include_neighbors: bool
        :return: The duplicate name, or ``None`` when creation fails.
        :rtype: str | None
        """
        parameters = dict(parameters or {})

        duplicate_parameters = self.get_duplicate_parameters(include_neighbors)
        space_parameters = duplicate_parameters.get("space_parameters", {})
        caller_space_parameters = parameters.get("space_parameters", {})
        space_parameters = {
            **space_parameters,
            **caller_space_parameters,
        }
        parameters["space_parameters"] = space_parameters


        if self.space and hasattr(self.space, "save_model"):
            try: 
                temporary_model = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
                temporary_model.close()
                success, model_path = self.space.save_model(temporary_model.name)
                if success:
                    space_parameters["model_file"] = model_path
                else:
                    self.get_logger().error("Failed to save model for duplication.")

            except Exception as e:
                self.get_logger().error(f"Exception occurred while saving model: {e}")
                Path(temporary_model.name).unlink()

        parameters["space_parameters"] = space_parameters

        name = await super().duplicate_node(name=name, parameters=parameters, include_neighbors=include_neighbors)

        # Load points into the space of the P-Node
        if name and self.space and self.space.size > 0:
            service_name = f"pnode/{name}/load_space"
            if service_name not in self.node_clients.keys():
                self.node_clients[service_name] = ServiceClientAsync(self, LoadSpace, service_name, callback_group=self.cbgroup_client)
            await self.node_clients[service_name].send_request_async(space=self.space.to_msg())
        Path(temporary_model.name).unlink()
        
        return name

def main(args = None):
    rclpy.init(args=args)

    pnode = PNode()

    rclpy.spin(pnode)

    pnode.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()