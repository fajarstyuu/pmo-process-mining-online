from dataclasses import dataclass, field
from typing import Any, List, Dict

@dataclass
class Statistic:
    events: List[Dict[str, Any]] = field(default_factory=list)
    case: List[Dict[str, Any]] = field(default_factory=list)
    variants: List[Dict[str, Any]] = field(default_factory=list)
    resources: List[Dict[str, Any]] = field(default_factory=list)
    general: List[Dict[str, Any]] = field(default_factory=list)

    def set_events(self, events: List[Dict[str, Any]]) -> None:
        self.events = events
    
    def set_case(self, case: List[Dict[str, Any]]) -> None:
        self.case = case

    def set_variants(self, variants: List[Dict[str, Any]]) -> None:
        self.variants = variants

    def set_resources(self, resources: List[Dict[str, Any]]) -> None:
        self.resources = resources

    def set_general(self, general: List[Dict[str, Any]]) -> None:
        self.general = general

    def get_events(self) -> List[Dict[str, Any]]:
        return self.events
    
    def get_case(self) -> List[Dict[str, Any]]:
        return self.case

    def get_variants(self) -> List[Dict[str, Any]]:
        return self.variants
    
    def get_resources(self) -> List[Dict[str, Any]]:
        return self.resources
    
    def get_general(self) -> List[Dict[str, Any]]:
        return self.general

    def set_all(
        self,
        events: List[Dict[str, Any]],
        case: List[Dict[str, Any]],
        variants: List[Dict[str, Any]],
        resources: List[Dict[str, Any]],
        general: List[Dict[str, Any]],
    ) -> None:
        self.set_events(events)
        self.set_case(case)
        self.set_variants(variants)
        self.set_resources(resources)
        self.set_general(general)
