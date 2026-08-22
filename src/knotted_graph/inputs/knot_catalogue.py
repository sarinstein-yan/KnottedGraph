"""Small verified braid catalogue used by :class:`KnotFunction`.

The catalogue is intentionally modest: arbitrary user-supplied Artin braid
words are supported by the generic compiler, so the package does not need a
large hard-coded table to claim arbitrary-braid functionality.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KnotCatalogueEntry:
    canonical_name: str
    braid_word: tuple[int, ...]
    strands: int
    components: int
    aliases: tuple[str, ...] = ()
    torus_params: tuple[int, int] | None = None
    reference_field: str | None = None


_ENTRIES = (
    KnotCatalogueEntry("0_1", (), 1, 1, ("unknot", "trivial_knot"), (1, 1)),
    KnotCatalogueEntry("3_1", (1, 1, 1), 2, 1, ("trefoil", "trefoil_knot"), (2, 3)),
    KnotCatalogueEntry(
        "4_1",
        (1, -2, 1, -2),
        3,
        1,
        ("figure8", "figure_eight", "figure-eight", "figure_eight_knot"),
        None,
        "rudolph_figure_eight",
    ),
    KnotCatalogueEntry("5_1", (1, 1, 1, 1, 1), 2, 1, ("cinquefoil",), (2, 5)),
    KnotCatalogueEntry("hopf_link", (1, 1), 2, 2, ("hopf", "2_1^2"), (2, 2)),
    KnotCatalogueEntry(
        "solomon_link", (1, 1, 1, 1), 2, 2, ("solomon", "4_1^2"), (2, 4)
    ),
    KnotCatalogueEntry(
        "borromean_rings",
        (1, -2, 1, -2, 1, -2),
        3,
        3,
        ("borromean", "6_2^3"),
    ),
)


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


_LOOKUP: dict[str, KnotCatalogueEntry] = {}
for _entry in _ENTRIES:
    for _name in (_entry.canonical_name, *_entry.aliases):
        _LOOKUP[_normalize_name(_name)] = _entry


def get_knot_entry(name: str) -> KnotCatalogueEntry:
    """Return a built-in braid representative for ``name``."""
    key = _normalize_name(name)
    try:
        return _LOOKUP[key]
    except KeyError as exc:
        choices = ", ".join(entry.canonical_name for entry in _ENTRIES)
        raise KeyError(
            f"unknown built-in knot/link name {name!r}; available canonical names: {choices}. "
            "Use KnotFunction.from_braid(...) for arbitrary braid closures."
        ) from exc


def available_knot_names() -> tuple[str, ...]:
    return tuple(entry.canonical_name for entry in _ENTRIES)


__all__ = ["KnotCatalogueEntry", "available_knot_names", "get_knot_entry"]
