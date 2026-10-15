"""Modele de donnees commun a tous les referentiels."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Link:
    """Lien de tracabilite declare dans DOORS."""
    direction: str          # "In" | "Out" | "Attr"
    link_type: str          # Specifies, Derives, Evidence, ...
    target_module: str      # nom de module DOORS tel qu'exporte
    target_abs: Optional[int] = None   # numero absolu (#303)
    target_id: Optional[str] = None    # ID resolu si le module est charge
    source: str = "ObjectLinks"        # provenance (ObjectLinks / ParentRequirements / ...)


@dataclass
class Requirement:
    uid: str                      # identifiant unique interne : "<module>::<ID>"
    rid: str                      # ID DOORS (ex. "SA.10.RS 1014")
    abs_no: Optional[int]         # numero absolu DOORS
    module: str                   # nom de module (ModuleRich/Name)
    level: str                    # CUS / RS / AD / TPS / EL
    text: str
    obj_type: str
    number: str = ""              # Object Number (1.1.0-2)
    section: str = ""             # chemin de section reconstruit
    allocated_to: str = ""
    scope: str = ""
    rationale: str = ""
    alloc_status: str = ""
    demo: str = ""
    links: List[Link] = field(default_factory=list)
    row: int = 0

    @property
    def is_requirement(self) -> bool:
        return self.obj_type in _REQ_TYPES

    @property
    def is_heading(self) -> bool:
        return self.obj_type == "Heading"


_REQ_TYPES: set = set()


def set_requirement_types(types) -> None:
    """Configure globalement les ObjectType consideres comme exigences."""
    global _REQ_TYPES
    _REQ_TYPES = set(types)


@dataclass
class Module:
    name: str                     # ModuleRich/Name  (ex. "Doors and entrances.RS")
    prefix: str                   # ModuleRich/Prefix (ex. "SA.10.RS")
    level: str
    path: str                     # Module Path DOORS
    baseline: str
    export_time: str
    file: str
    requirements: List[Requirement] = field(default_factory=list)
    meta: Dict[str, str] = field(default_factory=dict)

    def by_abs(self) -> Dict[int, Requirement]:
        return {r.abs_no: r for r in self.requirements if r.abs_no is not None}


@dataclass
class Corpus:
    modules: List[Module] = field(default_factory=list)

    def by_level(self, level: str) -> List[Module]:
        return [m for m in self.modules if m.level == level]

    def requirements(self, level: Optional[str] = None) -> List[Requirement]:
        out: List[Requirement] = []
        for m in self.modules:
            if level is None or m.level == level:
                out.extend(m.requirements)
        return out

    def module_index(self) -> Dict[str, Module]:
        """Index tolerant : nom complet, nom sans extension, prefixe."""
        idx: Dict[str, Module] = {}
        for m in self.modules:
            for key in {m.name, m.name.lower(), m.prefix, m.prefix.lower(),
                        m.name.rsplit(".", 1)[0].lower()}:
                if key:
                    idx.setdefault(key, m)
        return idx
