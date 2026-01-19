from typing import Any, Dict, List, Optional

from process_mining.model.Statistic import Statistic


class StatisticNotLoadedError(RuntimeError):

    """Raised when a Statistic instance is requested before being set."""


class StatisticModelService:

    """Lightweight helper that stores and exposes Statistic data."""

    def __init__(self, statistic: Optional[Statistic] = None) -> None:
        self._statistic = statistic


    def set_statistic_data(
        self,
    events: Optional[List[Dict[str, Any]]] = None,
    case: Optional[List[Dict[str, Any]]] = None,
    variants: Optional[List[Dict[str, Any]]] = None,
    resources: Optional[List[Dict[str, Any]]] = None,
    general: Optional[List[Dict[str, Any]]] = None,
    ) -> Statistic:
        if self._statistic is None:
            raise StatisticNotLoadedError("Statistic has not been set")
        self._statistic.set_all(
            events=events,
            case=case,
            variants=variants,
            resources=resources,
            general=general,
        )

    def get_statistic_data(self) -> Dict[str, List[Dict[str, Any]]]:
        if self._statistic is None:
            raise StatisticNotLoadedError("Statistic has not been set")
        return {
            'events': self._statistic.get_events(),
            'case': self._statistic.get_case(),
            'variants': self._statistic.get_variants(),
            'resources': self._statistic.get_resources(),
            'general': self._statistic.get_general(),
        }
