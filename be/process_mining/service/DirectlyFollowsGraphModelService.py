from process_mining.model.DirectlyFollowsGraph import DirectlyFollowsGraph
from typing import Any, List, Dict, Optional
import pandas as pd

class DirectlyFollowsGraphModelService:
    """Simple accessor for DirectlyFollowsGraph dataclasses."""

    def __init__(self, graph: Optional[DirectlyFollowsGraph] = None) -> None:
        self._graph = graph

    def set_nodes(self, nodes: List[Dict[str, Any]]) -> None:
        self._graph.set_nodes(nodes)

    def set_edges(self, edges: List[Dict[str, Any]]) -> None:
        self._graph.set_edges(edges)

    def set_data(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> None:
        self._graph.set_data(nodes, edges)

    def get_nodes(self) -> List[Dict[str, Any]]:
        return self._graph.get_nodes()

    def get_edges(self) -> List[Dict[str, Any]]:
        return self._graph.get_edges()

    def set_graph(self, graph: DirectlyFollowsGraph) -> None:
        if graph is None:
            raise ValueError("graph cannot be None")
        self._graph = graph
