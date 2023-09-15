import math, collections, copy
import numpy as np
from anastruct.basic import FEMException, arg_to_list
from anastruct.fem.postprocess import SystemLevel as post_sl
from anastruct.fem.elements import Element
from anastruct.vertex import Vertex
from anastruct.fem import plotter
from anastruct.sectionbase import properties
from anastruct.fem import system_components

from typing import (
    TYPE_CHECKING,
    Any,
    Collection,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from anastruct.fem.node import Node

Spring = Dict[int, float]
MpType = Dict[int, float]

class SystemElements:

    def __init__(
            self,
            figsize: Tuple[float, float] = (12, 8),
            EA: float = 15e3,
            EI: float = 5e3,
            load_factor: float = 1.0,
            mesh: int = 50,
        ):
            """
            * E = Young's modulus
            * A = Area
            * I = Moment of Inertia

            :param figsize: Set the standard plotting size.
            :param EA: Standard E * A. Set the standard values of EA if none provided when
                    generating an element.
            :param EI: Standard E * I. Set the standard values of EA if none provided when
                    generating an element.
            :param load_factor: Multiply all loads with this factor.
            :param mesh: Plotting mesh. Has no influence on the calculation.
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
            self.element_map: Dict[
                int, Element
            ] = {}  # maps element ids to the Element objects.
            self.node_map: Dict[
                int, Node  # pylint: disable=used-before-assignment
            ] = {}  # maps node ids to the Node objects.
            self.node_element_map: Dict[
                int, List[Element]
            ] = {}  # maps node ids to Element objects
            # keys matrix index (for both row and columns), values K, are processed
            # assemble_system_matrix
            self.system_spring_map: Dict[int, float] = {}

            # list of indexes that remain after conditions are applied
            self._remainder_indexes: List[int] = []

            # keep track of the nodes of the supports
            self.supports_fixed: List[Node] = []
            self.supports_hinged: List[Node] = []
            self.supports_rotational: List[Node] = []
            self.internal_hinges: List[Node] = []
            self.supports_roll: List[Node] = []
            self.supports_spring_x: List[Tuple[Node, bool]] = []
            self.supports_spring_z: List[Tuple[Node, bool]] = []
            self.supports_spring_y: List[Tuple[Node, bool]] = []
            self.supports_roll_direction: List[int] = []
            self.inclined_roll: Dict[
                int, float
            ] = {}  # map node ids to inclination angle relative to global x-axis.
            self.supports_roll_rotate: List[bool] = []

            # save tuples of the arguments for copying purposes.
            self.supports_spring_args: List[tuple] = []

            # keep track of the loads
            self.loads_point: Dict[
                int, Tuple[float, float]
            ] = {}  # node ids with a point loads {node_id: (x, y)}
            self.loads_q: Dict[
                int, List[Tuple[float, float]]
            ] = {}  # element ids with a q-loadad
            self.loads_moment: Dict[int, float] = {}
            self.loads_dead_load: Set[
                int
            ] = set()  # element ids with q-load due to dead load

            # results
            self.reaction_forces: Dict[int, Node] = {}  # node objects
            self.non_linear = False
            self.non_linear_elements: Dict[
                int, Dict[int, float]
            ] = (
                {}
            )  # keys are element ids, values are dicts: {node_index: max moment capacity}
            self.buckling_factor: Optional[float] = None

            # previous point of element
            self._previous_point = Vertex(0, 0)
            self.load_factor = load_factor

            # Objects state
            self.count = 0
            self.system_matrix: Optional[np.ndarray] = None
            self.system_force_vector: Optional[np.ndarray] = None
            self.system_displacement_vector: Optional[np.ndarray] = None
            self.shape_system_matrix: Optional[
                int
            ] = None  # actually is the size of the square system matrix
            self.reduced_force_vector: Optional[np.ndarray] = None
            self.reduced_system_matrix: Optional[np.ndarray] = None
            self._vertices: Dict[Vertex, int] = {}  # maps vertices to node ids

    # @property
    # def id_last_element(self) -> int:
    #     return max(self.element_map.keys())

    # @property
    # def id_last_node(self) -> int:
    #     return max(self.node_map.keys())

    def add_element(
            self,
            location: Union[
                Sequence[Sequence[float]], Sequence[Vertex], Sequence[float], Vertex
            ],
            EA: Optional[float] = None,
            EI: Optional[float] = None,
            g: float = 0,
            mp: Optional[MpType] = None,
            spring: Optional[Spring] = None,
            **kwargs: Any,
        ) -> int:
            """
            :param location: The two nodes of the element or the next node of the element.

                :Example:

                .. code-block:: python

                    location=[[x, y], [x, y]]
                    location=[Vertex, Vertex]
                    location=[x, y]
                    location=Vertex

            :param EA: EA
            :param EI: EI
            :param g: Weight per meter. [kN/m] / [N/m]
            :param mp: Set a maximum plastic moment capacity. Keys are integers representing
                    the nodes. Values are the bending moment capacity.

                :Example:

                .. code-block:: python

                    mp={1: 210e3,
                        2: 180e3}

            :param spring: Set a rotational spring or a hinge (k=0) at node 1 or node 2.

                :Example:

                .. code-block:: python

                    spring={1: k
                            2: k}


                    # Set a hinged node:
                    spring={1: 0}


            :return: Elements ID.
            """

            if mp is None:
                mp = {}
            if spring is None:
                spring = {}

            element_type = kwargs.get("element_type", "general")

            EA = self.EA if EA is None else EA
            EI = self.EI if EI is None else EI

            section_name = ""
            # change EA EI and g if steel section specified
            if "steelsection" in kwargs:
                section_name, EA, EI, g = properties.steel_section_properties(**kwargs)
            # change EA EI and g if rectangle section specified
            if "h" in kwargs:
                section_name, EA, EI, g = properties.rectangle_properties(**kwargs)
            # change EA EI and g if circle section specified
            if "d" in kwargs:
                section_name, EA, EI, g = properties.circle_properties(**kwargs)

            if element_type == "truss":
                EI = 1e-14

            # add the element number
            self.count += 1

            point_1, point_2 = system_components.util.det_vertices(self, location)
            node_id1, node_id2 = system_components.util.det_node_ids(self, point_1, point_2)

            (
                point_1,
                point_2,
                node_id1,
                node_id2,
                spring,
                mp,
                angle,
            ) = system_components.util.force_elements_orientation(
                point_1, point_2, node_id1, node_id2, spring, mp
            )

            system_components.util.append_node_id(
                self, point_1, point_2, node_id1, node_id2
            )

            # add element
            element = Element(
                id_=self.count,
                EA=EA,
                EI=EI,
                l=(point_2 - point_1).modulus(),
                angle=angle,
                vertex_1=point_1,
                vertex_2=point_2,
                type_=element_type,
                spring=spring,
                section_name=section_name,
            )
            element.node_id1 = node_id1
            element.node_id2 = node_id2
            element.node_map = {
                node_id1: self.node_map[node_id1],
                node_id2: self.node_map[node_id2],
            }

            self.element_map[self.count] = element

            for node in (node_id1, node_id2):
                if node in self.node_element_map:
                    self.node_element_map[node].append(element)
                else:
                    self.node_element_map[node] = [element]

            # Register the elements per node
            for node_id in (node_id1, node_id2):
                self.node_map[node_id].elements[element.id] = element

            assert mp is not None
            if len(mp) > 0:
                assert isinstance(mp, dict), "The mp parameter should be a dictionary."
                self.non_linear_elements[element.id] = mp
                self.non_linear = True
            system_components.assembly.dead_load(self, g, element.id)

            return self.count

    def solve(
        self,
        force_linear: bool = False,
        verbosity: int = 0,
        max_iter: int = 200,
        geometrical_non_linear: int = False,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Compute the results of current model.

        :param force_linear: Force a linear calculation. Even when the system has non linear nodes.
        :param verbosity: 0. Log calculation outputs. 1. silence.
        :param max_iter: Maximum allowed iterations.
        :param geometrical_non_linear: Calculate second order effects and determine the
                                        buckling factor.
        :return: Displacements vector.


        Development **kwargs:
            :param naked: Whether or not to run the solve function without doing post processing.
            :param discretize_kwargs: When doing a geometric non linear analysis you can reduce or
                                        increase the number of elements created that are used for
                                        determining the buckling_factor
        """

        # kwargs: arguments for the iterative solver callers such as the _stiffness_adaptation
        #         method.
        #                naked (bool) Default = False, if True force lines won't be computed.

        for node_id in self.node_map:
            system_components.util.check_internal_hinges(self, node_id)

        if self.system_displacement_vector is None:
            system_components.assembly.process_supports(self)
            assert self.system_displacement_vector is not None

        naked = kwargs.get("naked", False)

        if not naked:
            if not self.validate():
                if all(
                    ["general" in element.type for element in self.element_map.values()]
                ):
                    raise FEMException(
                        "StabilityError",
                        "The eigenvalues of the stiffness matrix are non zero, "
                        "which indicates a instable structure. "
                        "Check your support conditions",
                    )

        # (Re)set force vectors
        for el in self.element_map.values():
            el.reset()
        system_components.assembly.prep_matrix_forces(self)
        assert (
            self.system_force_vector is not None
        ), "There are no forces on the structure"

        if self.non_linear and not force_linear:
            return system_components.solver.stiffness_adaptation(
                self, verbosity, max_iter
            )

        system_components.assembly.assemble_system_matrix(self)
        if geometrical_non_linear:
            discretize_kwargs = kwargs.get("discretize_kwargs", None)
            self.buckling_factor = system_components.solver.geometrically_non_linear(
                self,
                verbosity,
                return_buckling_factor=True,
                discretize_kwargs=discretize_kwargs,
            )
            return self.system_displacement_vector

        system_components.assembly.process_conditions(self)

        # solution of the reduced system (reduced due to support conditions)
        assert self.reduced_system_matrix is not None
        assert self.reduced_force_vector is not None
        reduced_displacement_vector = np.linalg.solve(
            self.reduced_system_matrix, self.reduced_force_vector
        )

        # add the solution of the reduced system in the complete system displacement vector
        assert self.shape_system_matrix is not None
        self.system_displacement_vector = np.zeros(self.shape_system_matrix)
        np.put(
            self.system_displacement_vector,
            self._remainder_indexes,
            reduced_displacement_vector,
        )

        # determine the displacement vector of the elements
        for el in self.element_map.values():
            index_node_1 = (el.node_1.id - 1) * 3
            index_node_2 = (el.node_2.id - 1) * 3

            # node 1 ux, uz, phi
            el.element_displacement_vector[:3] = self.system_displacement_vector[
                index_node_1 : index_node_1 + 3
            ]
            # node 2 ux, uz, phi
            el.element_displacement_vector[3:] = self.system_displacement_vector[
                index_node_2 : index_node_2 + 3
            ]
            el.determine_force_vector()

        if not naked:
            # determining the node results in post processing class
            self.post_processor.node_results_elements()
            self.post_processor.node_results_system()
            self.post_processor.reaction_forces()
            self.post_processor.element_results()

            # check the values in the displacement vector for extreme values, indicating a
            # flawed calculation
            assert np.any(self.system_displacement_vector < 1e6), (
                "The displacements of the structure exceed 1e6. "
                "Check your support conditions,"
                "or your elements Young's modulus"
            )

        return self.system_displacement_vector

    def validate(self, min_eigen: float = 1e-9) -> bool:
        """
        Validate the stability of the stiffness matrix.

        :param min_eigen: Minimum value of the eigenvalues of the stiffness matrix. This value
        should be close to zero.
        """

        ss = copy.copy(self)
        system_components.assembly.prep_matrix_forces(ss)
        assert (
            np.abs(ss.system_force_vector).sum() != 0
        ), "There are no forces on the structure"
        ss._remainder_indexes = []
        system_components.assembly.assemble_system_matrix(ss)

        system_components.assembly.process_conditions(ss)

        w, _ = np.linalg.eig(ss.reduced_system_matrix)
        return bool(np.all(w > min_eigen))

    def add_support_hinged(self, node_id: Union[int, Sequence[int]]) -> None:
        """
        Model a hinged support at a given node.

        :param node_id: Represents the nodes ID
        """
        if not isinstance(node_id, collections.abc.Iterable):
            node_id = [node_id]

        for id_ in node_id:
            id_ = _negative_index_to_id(id_, self.node_map.keys())

            # add the support to the support list for the plotter
            self.supports_hinged.append(self.node_map[id_])

    def add_support_roll(
        self,
        node_id: Union[Sequence[int], int],
        direction: Union[Sequence[Union[str, int]], Union[str, int]] = "x",
        angle: Union[Sequence[Optional[float]], Optional[float]] = None,
        rotate: Union[Sequence[bool], bool] = True,
    ) -> None:
        """
        Adds a rolling support at a given node.

        :param node_id: Represents the nodes ID
        :param direction: Represents the direction that is free: 'x', 'y'
        :param angle: Angle in degrees relative to global x-axis.
                                If angle is given, the support will be inclined.
        :param rotate: If set to False, rotation at the roller will also be restrained.
        """
        if not isinstance(node_id, collections.abc.Iterable):
            node_id = [node_id]
        if not isinstance(direction, collections.abc.Iterable):
            direction = [direction]
        if not isinstance(angle, collections.abc.Iterable):
            angle = [angle]
        if not isinstance(rotate, collections.abc.Iterable):
            rotate = [rotate]

        assert len(node_id) == len(direction) == len(angle) == len(rotate)

        for id_, direction_, angle_, rotate_ in zip(node_id, direction, angle, rotate):
            id_ = _negative_index_to_id(id_, self.node_map.keys())

            if direction_ == "x":
                direction_i = 2
            elif direction_ == "y":
                direction_i = 1
            else:
                direction_i = int(direction_)

            if angle_ is not None:
                direction_i = 2
                self.inclined_roll[id_] = float(np.radians(-angle_))

            # add the support to the support list for the plotter
            self.supports_roll.append(self.node_map[id_])
            self.supports_roll_direction.append(direction_i)
            self.supports_roll_rotate.append(rotate_)

    def add_support_fixed(
        self,
        node_id: Union[Sequence[int], int],
    ) -> None:
        """
        Add a fixed support at a given node.

        :param node_id: Represents the nodes ID
        """
        if not isinstance(node_id, collections.abc.Iterable):
            node_id = [
                node_id,
            ]

        for id_ in node_id:
            id_ = _negative_index_to_id(id_, self.node_map.keys())
            system_components.util.support_check(self, id_)

            # add the support to the support list for the plotter
            self.supports_fixed.append(self.node_map[id_])

    def q_load(
        self,
        q: Union[float, Sequence[float]],
        element_id: Union[int, Sequence[int]],
        direction: Union[str, Sequence[str]] = "element",
        rotation: Optional[Union[float, Sequence[float]]] = None,
        q_perp: Union[float, Sequence[float]] = None,
    ) -> None:
        """
        Apply a q-load to an element.

        :param element_id: representing the element ID
        :param q: value of the q-load
        :param direction: "element", "x", "y", "parallel"
        :param rotation: Rotate the force clockwise. Rotation is in degrees
        :param q_perp: value of any q-load perpendicular to the indication direction/rotation
        """
        q_arr: Sequence[Sequence[float]]
        q_perp_arr: Sequence[Sequence[float]]
        if isinstance(q, Sequence):
            q_arr = [q]
        elif isinstance(q, (int, float)):
            q_arr = [[q, q]]
        if q_perp is None:
            q_perp_arr = [[0, 0]]
        elif isinstance(q_perp, Sequence):
            q_perp_arr = [q_perp]
        elif isinstance(q_perp, (int, float)):
            q_perp_arr = [[q_perp, q_perp]]

        if rotation is None:
            direction_flag = True
        else:
            direction_flag = False

        n_elems = len(element_id) if isinstance(element_id, Sequence) else 1
        element_id = arg_to_list(element_id, n_elems)
        direction = arg_to_list(direction, n_elems)
        rotation = arg_to_list(rotation, n_elems)
        q_arr = arg_to_list(q_arr, n_elems)
        q_perp_arr = arg_to_list(q_perp_arr, n_elems)

        for i, element_idi in enumerate(element_id):
            id_ = _negative_index_to_id(element_idi, self.element_map.keys())
            self.plotter.max_q = max(
                self.plotter.max_q,
                (q_arr[i][0] ** 2 + q_perp_arr[i][0] ** 2) ** 0.5,
                (q_arr[i][1] ** 2 + q_perp_arr[i][1] ** 2) ** 0.5,
            )

            if direction_flag:
                if direction[i] == "x":
                    rotation[i] = 0
                elif direction[i] == "y":
                    rotation[i] = np.pi / 2
                elif direction[i] == "parallel":
                    rotation[i] = self.element_map[element_id[i]].angle
                else:
                    rotation[i] = np.pi / 2 + self.element_map[element_id[i]].angle
            else:
                rotation[i] = math.radians(rotation[i])
                direction[i] = "angle"

            cos = math.cos(rotation[i])
            sin = math.sin(rotation[i])
            self.loads_q[id_] = [
                (
                    (q_perp_arr[i][0] * cos + q_arr[i][0] * sin) * self.load_factor,
                    (q_arr[i][0] * self.orientation_cs * cos + q_perp_arr[i][0] * sin)
                    * self.load_factor,
                ),
                (
                    (q_perp_arr[i][1] * cos + q_arr[i][1] * sin) * self.load_factor,
                    (q_arr[i][1] * self.orientation_cs * cos + q_perp_arr[i][1] * sin)
                    * self.load_factor,
                ),
            ]
            el = self.element_map[id_]
            el.q_load = (
                self.orientation_cs * self.load_factor * q_arr[i][0],
                self.orientation_cs * self.load_factor * q_arr[i][1],
            )
            el.q_perp_load = (
                q_perp_arr[i][0] * self.load_factor,
                q_perp_arr[i][1] * self.load_factor,
            )
            el.q_direction = direction[i]
            el.q_angle = rotation[i]

    def point_load(
        self,
        node_id: Union[int, Sequence[int]],
        Fx: Union[float, Sequence[float]] = 0.0,
        Fy: Union[float, Sequence[float]] = 0.0,
        rotation: Union[float, Sequence[float]] = 0.0,
    ) -> None:
        """
        Apply a point load to a node.

        :param node_id: Nodes ID.
        :param Fx: Force in global x direction.
        :param Fy: Force in global x direction.
        :param rotation: Rotate the force clockwise. Rotation is in degrees.
        """
        n = len(node_id) if isinstance(node_id, Sequence) else 1
        node_id = arg_to_list(node_id, n)
        Fx = arg_to_list(Fx, n)
        Fy = arg_to_list(Fy, n)
        rotation = arg_to_list(rotation, n)

        for i, node_idi in enumerate(node_id):
            id_ = _negative_index_to_id(node_idi, self.node_map.keys())
            if (
                id_ in self.inclined_roll
                and np.mod(self.inclined_roll[id_], np.pi / 2) != 0
            ):
                raise FEMException(
                    "StabilityError",
                    "Point loads may not be placed at the location of "
                    "inclined roller supports",
                )
            self.plotter.max_system_point_load = max(
                self.plotter.max_system_point_load, (Fx[i] ** 2 + Fy[i] ** 2) ** 0.5
            )
            cos = math.cos(math.radians(rotation[i]))
            sin = math.sin(math.radians(rotation[i]))
            self.loads_point[id_] = (
                (Fx[i] * cos + Fy[i] * sin) * self.load_factor,
                (Fy[i] * self.orientation_cs * cos + Fx[i] * sin) * self.load_factor,
            )

    def moment_load(
        self, node_id: Union[int, Sequence[int]], Ty: Union[float, Sequence[float]]
    ) -> None:
        """
        Apply a moment on a node.

        :param node_id: Nodes ID.
        :param Ty: Moments acting on the node.
        """
        n = len(node_id) if isinstance(node_id, Sequence) else 1
        node_id = arg_to_list(node_id, n)
        Ty = arg_to_list(Ty, n)

        for i, node_idi in enumerate(node_id):
            id_ = _negative_index_to_id(node_idi, self.node_map.keys())
            self.loads_moment[id_] = Ty[i] * self.load_factor

    def show_structure(
        self,
        verbosity: int = 0,
        scale: float = 1.0,
        offset: Tuple[float, float] = (0, 0),
        figsize: Optional[Tuple[float, float]] = None,
        show: bool = True,
        supports: bool = True,
        values_only: bool = False,
        annotations: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Optional["Figure"]]:
        """
        Plot the structure.

        :param factor: Influence the plotting scale.
        :param verbosity:  0: All information, 1: Suppress information.
        :param scale: Scale of the plot.
        :param offset: Offset the plots location on the figure.
        :param figsize: Change the figure size.
        :param show: Plot the result or return a figure.
        :param values_only: Return the values that would be plotted as tuple containing
                            two arrays: (x, y)
        :param annotations: if True, structure annotations are plotted. It includes section name.
                                        Note: only works when verbosity is equal to 0.
        """
        figsize = self.figsize if figsize is None else figsize
        if values_only:
            return self.plot_values.structure()
        return self.plotter.plot_structure(
            figsize, verbosity, show, supports, scale, offset, annotations=annotations
        )

    def show_bending_moment(
        self,
        factor: Optional[float] = None,
        verbosity: int = 0,
        scale: float = 1,
        offset: Tuple[float, float] = (0, 0),
        figsize: Tuple[float, float] = None,
        show: bool = True,
        values_only: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Optional["Figure"]]:
        """
        Plot the bending moment.

        :param factor: Influence the plotting scale.
        :param verbosity:  0: All information, 1: Suppress information.
        :param scale: Scale of the plot.
        :param offset: Offset the plots location on the figure.
        :param figsize: Change the figure size.
        :param show: Plot the result or return a figure.
        :param values_only: Return the values that would be plotted as tuple containing
                            two arrays: (x, y)
        """
        if values_only:
            return self.plot_values.bending_moment(factor)
        figsize = self.figsize if figsize is None else figsize
        return self.plotter.bending_moment(
            factor, figsize, verbosity, scale, offset, show
        )

    def show_axial_force(
        self,
        factor: Optional[float] = None,
        verbosity: int = 0,
        scale: float = 1,
        offset: Tuple[float, float] = (0, 0),
        figsize: Optional[Tuple[float, float]] = None,
        show: bool = True,
        values_only: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Optional["Figure"]]:
        """
        Plot the axial force.

        :param factor: Influence the plotting scale.
        :param verbosity:  0: All information, 1: Suppress information.
        :param scale: Scale of the plot.
        :param offset: Offset the plots location on the figure.
        :param figsize: Change the figure size.
        :param show: Plot the result or return a figure.
        :param values_only: Return the values that would be plotted as tuple containing
                            two arrays: (x, y)
        """
        if values_only:
            return self.plot_values.axial_force(factor)
        figsize = self.figsize if figsize is None else figsize
        return self.plotter.axial_force(factor, figsize, verbosity, scale, offset, show)

    def show_shear_force(
        self,
        factor: Optional[float] = None,
        verbosity: int = 0,
        scale: float = 1,
        offset: Tuple[float, float] = (0, 0),
        figsize: Optional[Tuple[float, float]] = None,
        show: bool = True,
        values_only: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Optional["Figure"]]:
        """
        Plot the shear force.

        :param factor: Influence the plotting scale.
        :param verbosity:  0: All information, 1: Suppress information.
        :param scale: Scale of the plot.
        :param offset: Offset the plots location on the figure.
        :param figsize: Change the figure size.
        :param show: Plot the result or return a figure.
        :param values_only: Return the values that would be plotted as tuple containing
                            two arrays: (x, y)
        """
        if values_only:
            return self.plot_values.shear_force(factor)
        figsize = self.figsize if figsize is None else figsize
        return self.plotter.shear_force(factor, figsize, verbosity, scale, offset, show)

    def show_reaction_force(
        self,
        verbosity: int = 0,
        scale: float = 1,
        offset: Tuple[float, float] = (0, 0),
        figsize: Optional[Tuple[float, float]] = None,
        show: bool = True,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Optional["Figure"]]:
        """
        Plot the reaction force.

        :param verbosity: 0: All information, 1: Suppress information.
        :param scale: Scale of the plot.
        :param offset: Offset the plots location on the figure.
        :param figsize: Change the figure size.
        :param show: Plot the result or return a figure.
        """
        figsize = self.figsize if figsize is None else figsize
        return self.plotter.reaction_force(figsize, verbosity, scale, offset, show)

    def show_displacement(
        self,
        factor: Optional[float] = None,
        verbosity: int = 0,
        scale: float = 1,
        offset: Tuple[float, float] = (0, 0),
        figsize: Optional[Tuple[float, float]] = None,
        show: bool = True,
        linear: bool = False,
        values_only: bool = False,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Optional["Figure"]]:
        """
        Plot the displacement.

        :param factor: Influence the plotting scale.
        :param verbosity:  0: All information, 1: Suppress information.
        :param scale: Scale of the plot.
        :param offset: Offset the plots location on the figure.
        :param figsize: Change the figure size.
        :param show: Plot the result or return a figure.
        :param linear: Don't evaluate the displacement values in between the elements
        :param values_only: Return the values that would be plotted as tuple containing
                            two arrays: (x, y)
        """
        if values_only:
            return self.plot_values.displacements(factor, linear)
        figsize = self.figsize if figsize is None else figsize
        return self.plotter.displacements(
            factor, figsize, verbosity, scale, offset, show, linear
        )

def _negative_index_to_id(idx: int, collection: Collection[int]) -> int:
    if not isinstance(idx, int):
        if int(idx) == idx:  # if it can be non-destructively cast to an integer
            idx = int(idx)
        else:
            raise TypeError("Node or element id must be an integer")
    if idx > 0:
        return idx
    else:
        return max(collection) + (idx + 1)
