import math, re, collections, copy
import numpy as np
from anastruct.basic import FEMException, args_to_lists
from anastruct.fem.postprocess import SystemLevel as post_sl
from anastruct.fem.elements import Element
from anastruct.vertex import Vertex
from anastruct.fem import plotter
from anastruct.fem import system_components
from anastruct.vertex import vertex_range

class SystemElements:
    """
    Modelling any structure starts with an object of this class.

    :ivar EA: Standard axial stiffness of elements, default=15,000
    :ivar EI: Standard bending stiffness of elements, default=5,000
    :ivar figsize: (tpl) Matplotlibs standard figure size
    :ivar element_map: (dict) Keys are the element ids, values are the element objects
    :ivar node_map: (dict) Keys are the node ids, values are the node objects.
    :ivar node_element_map: (dict) maps node ids to element objects.
    :ivar loads_point: (dict) Maps node ids to point loads.
    :ivar loads_q: (dict) Maps element ids to q-loads.
    :ivar loads_moment: (dict) Maps node ids to moment loads.
    :ivar loads_dead_load: (set) Element ids that have a dead load applied.
    """

    def __init__(self, figsize=(12, 8), EA=15e3, EI=5e3, load_factor=1, mesh=50):
        """
        * E = Young's modulus
        * A = Area
        * I = Moment of Inertia

        :param figsize: (tpl) Set the standard plotting size.
        :param EA: (flt) Standard E * A. Set the standard values of EA if none provided when generating an element.
        :param EI: (flt) Standard E * I. Set the standard values of EA if none provided when generating an element.
        :param load_factor: (flt) Multiply all loads with this factor.
        :param mesh: (int) Plotting mesh. Has no influence on the calculation.
        """
        # init object
        self.post_processor = post_sl(self)
        self.plotter = plotter.Plotter(self, mesh)
        self.plot_values = plotter.PlottingValues(self, mesh)

        # standard values if none provided
        self.EA = EA
        self.EI = EI
        self.figsize = figsize
        self.orientation_cs = -1  # needed for the loads directions

        # structure system
        self.element_map = {}  # maps element ids to the Element objects.
        self.node_map = {}  # maps node ids to the Node objects.
        self.node_element_map = {}  # maps node ids to Element objects
        # keys matrix index (for both row and columns), values K, are processed assemble_system_matrix
        self.system_spring_map = {}

        # list of indexes that remain after conditions are applied
        self._remainder_indexes = []

        # keep track of the node_id of the supports
        self.supports_fixed = []
        self.supports_hinged = []
        self.supports_roll = []
        self.supports_spring_x = []
        self.supports_spring_z = []
        self.supports_spring_y = []
        self.supports_roll_direction = []
        self.inclined_roll = ({})

        # save tuples of the arguments for copying purposes.
        self.supports_spring_args = []

        # keep track of the loads
        self.loads_point = {}  # node ids with a point loads
        self.loads_q = {}  # element ids with a q-load
        self.loads_moment = {}
        self.loads_dead_load = set()  # element ids with q-load due to dead load

        # results
        self.reaction_forces = {}  # node objects
        self.non_linear = False
        self.non_linear_elements = {}  # keys are element ids, values are dicts: {node_index: max moment capacity}
        self.buckling_factor = None

        # previous point of element
        self._previous_point = Vertex(0, 0)
        self.load_factor = load_factor

        # Objects state
        self.count = 0
        self.system_matrix_locations = []
        self.system_matrix = None
        self.system_force_vector = None
        self.system_displacement_vector = None
        self.shape_system_matrix = None
        self.reduced_force_vector = None
        self.reduced_system_matrix = None
        self._vertices = {}  # maps vertices to node ids

    def add_element(self, location, EA=None, EI=None, g=0, mp=None, spring=None, **kwargs):
        """
        :param location: (list/ Vertex) The two nodes of the element or the next node of the element.

            :Example:

            .. code-block:: python

                   location=[[x, y], [x, y]]
                   location=[Vertex, Vertex]
                   location=[x, y]
                   location=Vertex

        :param EA: (flt) EA
        :param EI: (flt) EI
        :param g: (flt) Weight per meter. [kN/m] / [N/m]
        :param mp: (dict) Set a maximum plastic moment capacity. Keys are integers representing the nodes. Values
                          are the bending moment capacity.

            :Example:

            .. code-block:: python

                mp={1: 210e3,
                    2: 180e3}

        :param spring: (dict) Set a rotational spring or a hinge (k=0) at node 1 or node 2.

            :Example:

            .. code-block:: python

                spring={1: k
                        2: k}


                # Set a hinged node:
                spring={1: 0}


        :return: (int) Elements ID.
        """
        element_type = kwargs.get("element_type", "general")

        EA = self.EA if EA is None else EA
        EI = self.EI if EI is None else EI

        if element_type == 'truss':
            EI = 1e-14

        # add the element number
        self.count += 1

        point_1, point_2 = system_components.util.det_vertices(self, location)
        node_id1, node_id2 = system_components.util.det_node_ids(self, point_1, point_2)

        point_1, point_2, node_id1, node_id2, spring, mp, ai = \
            system_components.util.force_elements_orientation(point_1, point_2, node_id1, node_id2, spring, mp)

        system_components.util.append_node_id(self, point_1, point_2, node_id1, node_id2)
        system_components.util.ensure_single_hinge(self, spring, node_id1, node_id2)

        # add element
        element = Element(self.count, EA, EI, (point_2 - point_1).modulus(), ai, point_1, point_2, spring)
        element.node_id1 = node_id1
        element.node_id2 = node_id2
        element.node_map = {node_id1: self.node_map[node_id1],
                            node_id2: self.node_map[node_id2]}

        element.type = element_type

        self.element_map[self.count] = element

        for node in (node_id1, node_id2):
            if node in self.node_element_map:
                self.node_element_map[node].append(element)
            else:
                self.node_element_map[node] = [element]

        # Register the elements per node
        for node_id in (node_id1, node_id2):
            self.node_map[node_id].elements[element.id] = element

        if mp is not None:
            assert type(mp) == dict, "The mp parameter should be a dictionary."
            self.non_linear_elements[element.id] = mp
            self.non_linear = True
        system_components.assembly.dead_load(self, g, element.id)

        return self.count

    def solve(self, force_linear=False, verbosity=0, max_iter=200, geometrical_non_linear=False, **kwargs):

        """
        Compute the results of current model.

        :param force_linear: (bool) Force a linear calculation. Even when the system has non linear nodes.
        :param verbosity: (int) 0. Log calculation outputs. 1. silence.
        :param max_iter: (int) Maximum allowed iterations.
        :param geometrical_non_linear: (bool) Calculate second order effects and determine the buckling factor.
        :return: (array) Displacements vector.


        Development **kwargs:
            :param naked: (bool) Whether or not to run the solve function without doing post processing.
            :param discretize_kwargs: When doing a geometric non linear analysis you can reduce or increase the number
                                      of elements created that are used for determining the buckling_factor
        """

        # kwargs: arguments for the iterative solver callers such as the _stiffness_adaptation method.
        #                naked (bool) Default = False, if True force lines won't be computed.

        if self.system_displacement_vector is None:
            system_components.assembly.process_supports(self)

        naked = kwargs.get("naked", False)

        if not naked:
            if not self.validate():
                if all(['general' in element.type for element in self.element_map.values()]):
                    raise FEMException('StabilityError', 'The eigenvalues of the stiffness matrix are non zero, '
                                                         'which indicates a instable structure. '
                                                         'Check your support conditions')

        # (Re)set force vectors
        for el in self.element_map.values():
            el.reset()
        system_components.assembly.prep_matrix_forces(self)
        assert (self.system_force_vector is not None), "There are no forces on the structure"

        if self.non_linear and not force_linear:
            return system_components.solver.stiffness_adaptation(self, verbosity, max_iter)

        system_components.assembly.assemble_system_matrix(self)
        if geometrical_non_linear:
            discretize_kwargs = kwargs.get('discretize_kwargs', None)
            self.buckling_factor = system_components.solver.geometrically_non_linear(self, verbosity,
                                                                                     discretize_kwargs=discretize_kwargs)
            return self.system_displacement_vector

        system_components.assembly.process_conditions(self)

        # solution of the reduced system (reduced due to support conditions)
        reduced_displacement_vector = np.linalg.solve(self.reduced_system_matrix, self.reduced_force_vector)

        # add the solution of the reduced system in the complete system displacement vector
        self.system_displacement_vector = np.zeros(self.shape_system_matrix)
        np.put(self.system_displacement_vector, self._remainder_indexes, reduced_displacement_vector)

        # determine the displacement vector of the elements
        for el in self.element_map.values():
            index_node_1 = (el.node_1.id - 1) * 3
            index_node_2 = (el.node_2.id - 1) * 3

            # node 1 ux, uz, phi
            el.element_displacement_vector[:3] = self.system_displacement_vector[index_node_1: index_node_1 + 3]
            # node 2 ux, uz, phi
            el.element_displacement_vector[3:] = self.system_displacement_vector[index_node_2: index_node_2 + 3]
            el.determine_force_vector()

        if not naked:
            # determining the node results in post processing class
            self.post_processor.node_results_elements()
            self.post_processor.node_results_system()
            self.post_processor.reaction_forces()
            self.post_processor.element_results()

            # check the values in the displacement vector for extreme values, indicating a flawed calculation
            assert (np.any(self.system_displacement_vector < 1e6)), "The displacements of the structure exceed 1e6. " \
                                                                    "Check your support conditions," \
                                                                    "or your elements Young's modulus"

        return self.system_displacement_vector

    def validate(self, min_eigen=1e-9):
        """
        Validate the stability of the stiffness matrix.
        :param min_eigen: (flt) Minimum value of the eigenvalues of the stiffness matrix. This value should be close
        to zero.
        :return: (bool)
        """

        ss = copy.copy(self)
        system_components.assembly.prep_matrix_forces(ss)
        assert (np.abs(ss.system_force_vector).sum() != 0), "There are no forces on the structure"
        ss._remainder_indexes = []
        system_components.assembly.assemble_system_matrix(ss)

        system_components.assembly.process_conditions(ss)

        w, _ = np.linalg.eig(ss.reduced_system_matrix)
        return np.all(w > min_eigen)

    def add_support_hinged(self, node_id):
        """
        Model a hinged support at a given node.

        :param node_id: (int/ list) Represents the nodes ID
        """
        if not isinstance(node_id, collections.Iterable):
            node_id = (node_id,)

        for id_ in node_id:
            id_ = _negative_index_to_id(id_, self.node_map.keys())
            system_components.util.support_check(self, id_)

            # add the support to the support list for the plotter
            self.supports_hinged.append(self.node_map[id_])

    def add_support_roll(self, node_id, direction='x', angle = None):
        """
        Adds a rolling support at a given node.

        :param node_id: (int/ list) Represents the nodes ID
        :param direction: (int/ list) Represents the direction that is fixed: x = 1, y = 2
        """
        if not isinstance(node_id, collections.Iterable):
            node_id = (node_id,)

        if direction == "x":
            direction = 2
        elif direction == "y":
            direction = 1

        if angle is not None:
            direction = 2
            self.inclined_roll[id_] = np.radians(-angle)

        for id_ in node_id:
            id_ = _negative_index_to_id(id_, self.node_map.keys())
            system_components.util.support_check(self, id_)

            # add the support to the support list for the plotter
            self.supports_roll.append(self.node_map[id_])
            self.supports_roll_direction.append(direction)

    def add_support_fixed(self, node_id):
        """
        Add a fixed support at a given node.

        :param node_id: (int/ list) Represents the nodes ID
        """
        if not isinstance(node_id, collections.Iterable):
            node_id = (node_id,)

        for id_ in node_id:
            id_ = _negative_index_to_id(id_, self.node_map.keys())
            system_components.util.support_check(self, id_)

            # add the support to the support list for the plotter
            self.supports_fixed.append(self.node_map[id_])

    def q_load(self, q, element_id, direction="element"):
        """
        Apply a q-load to an element.

        :param element_id: (int/ list) representing the element ID
        :param q: (flt) value of the q-load
        :param direction: (str) "element", "x", "y"
        """
        q, element_id, direction = args_to_lists(q, element_id, direction)

        for i in range(len(element_id)):
            id_ = _negative_index_to_id(element_id[i], self.element_map.keys())
            self.plotter.max_q = max(self.plotter.max_q, abs(q[i]))
            self.loads_q[id_] = q[i] * self.orientation_cs * self.load_factor
            el = self.element_map[id_]
            el.q_load = q[i] * self.orientation_cs * self.load_factor
            el.q_direction = direction[i]

    def point_load(self, node_id, Fx=0, Fy=0, rotation=0):
        """
        Apply a point load to a node.

        :param node_id: (int/ list) Nodes ID.
        :param Fx: (flt/ list) Force in global x direction.
        :param Fy: (flt/ list) Force in global x direction.
        :param rotation: (flt/ list) Rotate the force clockwise. Rotation is in degrees.
        """
        node_id, Fx, Fy, rotation = args_to_lists(node_id, Fx, Fy, rotation)

        for i in range(len(node_id)):
            id_ = _negative_index_to_id(node_id[i], self.node_map.keys())
            self.plotter.max_system_point_load = max(self.plotter.max_system_point_load,
                                                     (Fx[i] ** 2 + Fy[i] ** 2) ** 0.5)
            cos = math.cos(math.radians(rotation[i]))
            sin = math.sin(math.radians(rotation[i]))
            self.loads_point[id_] = (Fx[i] * cos + Fy[i] * sin, Fy[i] * self.orientation_cs * cos + Fx[i] * sin)

    def moment_load(self, node_id, Ty):
        """
        Apply a moment on a node.

        :param node_id: (int/ list) Nodes ID.
        :param Ty: (flt/ list) Moments acting on the node.
        """
        node_id, Ty = args_to_lists(node_id, Ty)

        for i in range(len(node_id)):
            id_ = _negative_index_to_id(node_id[i], self.node_map.keys())
            self.loads_moment[id_] = Ty[i]

    def show_structure(self, verbosity=0, scale=1., offset=(0, 0), figsize=None, show=True, supports=True,
                       values_only=False):
        """
        Plot the structure.

        :param verbosity: (int) 0: All information, 1: Suppress information.
        :param scale: (flt) Scale of the plot.
        :param offset: (tpl) Offset the plots location on the figure.
        :param figsize: (tpl) Change the figure size.
        :param show: (bool) Plot the result or return a figure.
        :param supports: (bool) Show the supports.
        :param values_only: (bool) Return the values that would be plotted as tuple containing two arrays: (x, y)
        :return: (figure)
        """
        figsize = self.figsize if figsize is None else figsize
        if values_only:
            return self.plot_values.structure()
        return self.plotter.plot_structure(figsize, verbosity, show, supports, scale, offset)

    def show_bending_moment(self, factor=None, verbosity=0, scale=1, offset=(0, 0), figsize=None, show=True,
                            values_only=False):
        """
        Plot the bending moment.

        :param factor: (flt) Influence the plotting scale.
        :param verbosity: (int) 0: All information, 1: Suppress information.
        :param scale: (flt) Scale of the plot.
        :param offset: (tpl) Offset the plots location on the figure.
        :param figsize: (tpl) Change the figure size.
        :param show: (bool) Plot the result or return a figure.
        :param values_only: (bool) Return the values that would be plotted as tuple containing two arrays: (x, y)
        :return: (figure)
        """
        if values_only:
            return self.plot_values.bending_moment(factor)
        figsize = self.figsize if figsize is None else figsize
        return self.plotter.bending_moment(factor, figsize, verbosity, scale, offset, show)

    def show_axial_force(self, factor=None, verbosity=0, scale=1, offset=(0, 0), figsize=None, show=True,
                         values_only=False):
        """
        Plot the axial force.

        :param factor: (flt) Influence the plotting scale.
        :param verbosity: (int) 0: All information, 1: Suppress information.
        :param scale: (flt) Scale of the plot.
        :param offset: (tpl) Offset the plots location on the figure.
        :param figsize: (tpl) Change the figure size.
        :param show: (bool) Plot the result or return a figure.
        :param values_only: (bool) Return the values that would be plotted as tuple containing two arrays: (x, y)
        :return: (figure)
        """
        if values_only:
            return self.plot_values.axial_force(factor)
        figsize = self.figsize if figsize is None else figsize
        return self.plotter.axial_force(factor, figsize, verbosity, scale, offset, show)

    def show_shear_force(self, factor=None, verbosity=0, scale=1, offset=(0, 0), figsize=None, show=True,
                         values_only=False):
        """
        Plot the shear force.
        :param factor: (flt) Influence the plotting scale.
        :param verbosity: (int) 0: All information, 1: Suppress information.
        :param scale: (flt) Scale of the plot.
        :param offset: (tpl) Offset the plots location on the figure.
        :param figsize: (tpl) Change the figure size.
        :param show: (bool) Plot the result or return a figure.
        :param values_only: (bool) Return the values that would be plotted as tuple containing two arrays: (x, y)
        :return: (figure)
        """
        if values_only:
            return self.plot_values.shear_force(factor)
        figsize = self.figsize if figsize is None else figsize
        return self.plotter.shear_force(factor, figsize, verbosity, scale, offset, show)

    def show_reaction_force(self, verbosity=0, scale=1, offset=(0, 0), figsize=None, show=True):
        """
        Plot the reaction force.

        :param verbosity: (int) 0: All information, 1: Suppress information.
        :param scale: (flt) Scale of the plot.
        :param offset: (tpl) Offset the plots location on the figure.
        :param figsize: (tpl) Change the figure size.
        :param show: (bool) Plot the result or return a figure.
        :return: (figure)
        """
        figsize = self.figsize if figsize is None else figsize
        return self.plotter.reaction_force(figsize, verbosity, scale, offset, show)

    def show_displacement(self, factor=None, verbosity=0, scale=1, offset=(0, 0), figsize=None, show=True,
                          linear=False, values_only=False):
        """
        Plot the displacement.

        :param factor: (flt) Influence the plotting scale.
        :param verbosity: (int) 0: All information, 1: Suppress information.
        :param scale: (flt) Scale of the plot.
        :param offset: (tpl) Offset the plots location on the figure.
        :param figsize: (tpl) Change the figure size.
        :param show: (bool) Plot the result or return a figure.
        :param linear: (bool) Don't evaluate the displacement values in between the elements
        :param values_only: (bool) Return the values that would be plotted as tuple containing two arrays: (x, y)
        :return: (figure)
        """
        if values_only:
            return self.plot_values.displacements(factor, linear)
        figsize = self.figsize if figsize is None else figsize
        return self.plotter.displacements(factor, figsize, verbosity, scale, offset, show, linear)

def _negative_index_to_id(idx, collection):
    if idx > 0:
        return idx
    else:
        return max(collection) + (idx + 1)


