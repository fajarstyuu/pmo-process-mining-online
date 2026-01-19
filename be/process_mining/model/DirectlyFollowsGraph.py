from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class DirectlyFollowsGraph:
    """Representation of a Directly-Follows Graph (DFG) with frequencies."""
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)

    def set_nodes(self, nodes: List[Dict[str, Any]]):
        self.nodes = nodes

    def set_edges(self, edges: List[Dict[str, Any]]):
        self.edges = edges

    def set_data(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]):
        self.nodes = nodes
        self.edges = edges

    def get_nodes(self) -> List[Dict[str, Any]]:
        return self.nodes
    
    def get_edges(self) -> List[Dict[str, Any]]:
        return self.edges
    
    def get_data(self) -> Dict[str, Any]:
        return self.get_edges(), self.get_nodes()
    