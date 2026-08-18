# airapi — Python DSL for spatial accelerators
#
# Structural and declaration names live here.
# Imperative compute/memory operations live in airapi.ops (import airapi.ops).

from airapi._core import (
    # Launch hierarchy
    launch,
    segment,
    herd,

    # Buffer and tensor declarations
    alloc,
    tensor,
    symbol,

    # Block floating-point types
    BlockType,
    Field,

    # Fabric properties (used in requires= and placement=)
    Scratchpad,
    Cascade,
    Adjacency,
    Broadcast,
    CacheDomain,

    # Placement constraints
    Disjoint,

    # Fabric descriptor (body parameter)
    Fabric,

    # Body variant capability gate (decorator)
    requires,

    # Token join (kept in airapi because it is control-flow, not compute)
    wait,

    # JIT compilation entry points
    jit,
    compile,
)

__all__ = [
    "launch", "segment", "herd",
    "alloc", "tensor", "symbol",
    "BlockType", "Field",
    "Scratchpad", "Cascade", "Adjacency", "Broadcast", "CacheDomain",
    "Disjoint", "Fabric",
    "requires",
    "wait",
    "jit", "compile",
]
