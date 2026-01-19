from dataclasses import dataclass, field
from typing import Any, List, Dict

@dataclass
class ProcessModel:
    _nodes: List[Dict[str, Any]] = field(default_factory=list)
    _edges: List[Dict[str, Any]] = field(default_factory=list)
    _model_statistics: Dict[str, Any] = field(default_factory=dict)
    _evaluation_metrics: Dict[str, Any] = field(default_factory=dict)

    def set_nodes(self, nodes: List[Dict[str, Any]]):
        self._nodes = nodes

    def set_edges(self, edges: List[Dict[str, Any]]):
        self._edges = edges

    def set_model_statistics(self, model_statistics: Dict[str, Any]):
        self._model_statistics = model_statistics

    def set_evaluation_metrics(self, evaluation_metrics: Dict[str, Any]):
        self._evaluation_metrics = evaluation_metrics

    def get_nodes(self) -> List[Dict[str, Any]]:
        return self._nodes

    def get_edges(self) -> List[Dict[str, Any]]:
        return self._edges

    def get_model_statistics(self) -> Dict[str, Any]:
        return self._model_statistics

    def get_evaluation_metrics(self) -> Dict[str, Any]:
        return self._evaluation_metrics

    