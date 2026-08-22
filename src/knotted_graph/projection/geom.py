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
    
    def add_incident_arc(self, arc_id: int, angle: float):
        """Add one half-edge incidence, preserving repeated arc IDs."""
        self.incident_arcs.append((arc_id, angle))
    
    @cached_property
    def _raw_ccw_ordered_arcs(self) -> List[int]:
        """Return the four crossing incidences counter-clockwise.

        A legitimate self-crossing can contain the same arc ID more than once:
        an arc may start and end at the same crossing.  The previous code
        silently dropped every such crossing, which made the computed diagram
        depend on where a crossing landed relative to the polyline sampling.
        Incidences, rather than unique arc IDs, are the relevant objects here.
        """
        assert len(self.incident_arcs) == 4, \
            "Crossing must have exactly 4 incidences."
        arc_ids, angles = zip(*self.incident_arcs)
        ccw_idx = np.argsort(angles)
        return [arc_ids[i] for i in ccw_idx]

    @cached_property
    def ccw_ordered_arcs(self) -> List[int]:
        raw_order = self._raw_ccw_ordered_arcs
        assert self._correctly_overstrand is not None, \
            "Overstranding information is not set."
        if not self._correctly_overstrand:
            raw_order = raw_order[1:] + raw_order[:1]
        return raw_order

    @cached_property
    def pd_code(self) -> str:
        """Return the crossing as PD code."""
        return f"X[{','.join(map(str, self.ccw_ordered_arcs))}]"


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
