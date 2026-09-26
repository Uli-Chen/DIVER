"""The exploration and exploitation actions used by DIVER."""
from dataclasses import dataclass
from enum import Enum

class ActionKind(str, Enum):
    GLOBAL = "global_discovery"
    INCIDENT = "incident_expansion"

@dataclass(frozen=True)
class QueryAction:
    kind: ActionKind
    anchor: str | None = None
