import numpy as np
from shapely import Point, LineString
from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Tuple, Any, ClassVar


__all__ = [
    "Vertex",
    "Crossing",
    "Arc",
]


@dataclass
class Vertex:
    """A vertex in the knot diagram."""
    id: int
    key: Any # Optional key from node
    point: Point
    incident_arcs: List[Tuple[int, float]] = field(default_factory=list)
    
    def add_incident_arc(self, arc_id: int, angle: float):
        """Add an incident arc with its angle."""
        self.incident_arcs.append((arc_id, angle))
    
    @cached_property
    def ccw_ordered_arcs(self) -> List[int]:
        """Return the incident arcs ordered counter-clockwise by angle."""
        if not self.incident_arcs:
            return []
        arc_ids, angles = zip(*self.incident_arcs)
        ccw_idx = np.argsort(angles)
        return [arc_ids[i] for i in ccw_idx]
    
    @cached_property
    def pd_code(self) -> str:
        """Return the vertex as PD code."""
        # This is different from crossings. Isolated vertices are meaningful.
        return f"V[{','.join(map(str, self.ccw_ordered_arcs))}]"


@dataclass
class Crossing:
    """A crossing in the knot diagram."""
    id: int
    point: Point
    incident_arcs: List[Tuple[int, float]] = field(default_factory=list)
    _correctly_overstrand: bool = field(default=None, init=False, repr=False)
    _transversality_tolerance: ClassVar[float] = 1e-10
    
    def add_incident_arc(self, arc_id: int, angle: float):
        """Add an incident half-edge and reject numerically tangent crossings.

        A generic planar-diagram crossing has four distinct local half-edge
        directions.  If two cyclically adjacent rays are indistinguishable to
        floating-point precision, the two projected strands are tangent (or
        effectively tangent) and their crossing combinatorics are not stable
        under an arbitrarily small perturbation.  Such a projection must be
        rejected and another rotation sampled rather than treated as a valid PD.

        The test is dimensionless: it uses angular separation only, so it does
        not depend on the spatial scale of the embedded graph.
        """
        self.incident_arcs.append((arc_id, float(angle)))
        if len(self.incident_arcs) == 4:
            angles = np.mod(
                np.sort(np.asarray([value for _, value in self.incident_arcs])),
                2.0 * np.pi,
            )
            angles.sort()
            cyclic = np.diff(np.r_[angles, angles[0] + 2.0 * np.pi])
            if float(cyclic.min()) <= self._transversality_tolerance:
                raise ValueError(
                    "Nongeneric projection: crossing strands are numerically tangent."
                )
    
    @cached_property
    def _raw_ccw_ordered_arcs(self) -> List[int]:
        """Return the incident arcs ordered counter-clockwise by angle."""
        assert len(self.incident_arcs) == 4, \
            "Crossing must have exactly 4 incidences."
        
        arc_ids, angles = zip(*self.incident_arcs)
        if len(set(arc_ids)) < 4:
            return []  # Trivial self-crossing
        
        # Sort by angle for counter-clockwise order
        ccw_idx = np.argsort(angles)
        return [arc_ids[i] for i in ccw_idx]

    @cached_property
    def ccw_ordered_arcs(self) -> List[int]:
        # Rotate if the overstranding is incorrect
        raw_order = self._raw_ccw_ordered_arcs
        if not raw_order:
            return [] # Trivial self-crossing

        assert self._correctly_overstrand is not None, \
            "Overstranding information is not set."
        if not self._correctly_overstrand:
            raw_order = raw_order[1:] + raw_order[:1]
        return raw_order

    @cached_property
    def pd_code(self) -> str:
        """Return the crossing as PD code."""
        arcs = self.ccw_ordered_arcs
        if not arcs:
            return ""
        return f"X[{','.join(map(str, arcs))}]"


@dataclass
class Arc:
    """An arc segment between vertices/crossings."""
    _id_counter: ClassVar[int] = 0
    edge_key: str
    line: LineString
    start_type: str
    start_id: int
    end_type: str
    end_id: int
    id: int = field(init=False)

    def __post_init__(self):
        """Assign a unique ID after the object is created."""
        self.id = Arc._id_counter
        Arc._id_counter += 1

    @classmethod
    def reset_counter(cls):
        """Resets the global counter, useful for multiple independent runs."""
        cls._id_counter = 0
