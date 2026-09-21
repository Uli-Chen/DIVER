"""Explicit query actions for learning-free tri-action extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from ..models import normalize_label


class ActionKind(str, Enum):
    """The three semantically distinct extraction operators."""

    GLOBAL = "global_discovery"
    INCIDENT = "incident_expansion"
    CLOSURE = "ego_closure"


@dataclass(frozen=True, order=True)
class EntityPair:
    """A canonical unordered endpoint pair used by Closure.

    Closure asks whether two neighbors have any direct relationship; the
    evidence, rather than the controller, determines the returned direction.
    """

    left: str
    right: str

    def __post_init__(self) -> None:
        endpoints = sorted((normalize_label(self.left), normalize_label(self.right)))
        if not endpoints[0] or not endpoints[1]:
            raise ValueError("Closure entity-pair endpoints must be non-empty")
        if endpoints[0] == endpoints[1]:
            raise ValueError("Closure entity-pair endpoints must be distinct")
        object.__setattr__(self, "left", endpoints[0])
        object.__setattr__(self, "right", endpoints[1])

    @classmethod
    def from_values(cls, left: str, right: str) -> "EntityPair":
        return cls(left, right)

    def contains_directed(self, source: str, target: str) -> bool:
        return self == EntityPair.from_values(source, target)

    def to_list(self) -> list[str]:
        return [self.left, self.right]


# Backward-compatible import name for external scripts.  New code should use
# EntityPair because direction is deliberately not part of Closure selection.
DirectedPair = EntityPair


@dataclass(frozen=True)
class QueryAction:
    """One immutable controller decision consumed by the prompt renderer."""

    kind: ActionKind
    topic: str | None = None
    anchor: str | None = None
    pairs: tuple[EntityPair, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "topic": self.topic,
            "anchor": self.anchor,
            "pairs": [pair.to_list() for pair in self.pairs],
            "metadata": dict(self.metadata),
        }
