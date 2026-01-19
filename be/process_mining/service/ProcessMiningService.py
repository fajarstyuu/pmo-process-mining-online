from typing import Any, Dict, Optional

import pandas as pd

from process_mining.model.ProcessMining import ProcessMining


class ProcessMiningNotLoadedError(RuntimeError):
    """Raised when no ProcessMining configuration is stored."""


class ProcessMiningService:
    """Minimal helper for reading/writing ProcessMining objects."""

    def __init__(self, process_mining: Optional[ProcessMining] = None) -> None:
        self._process_mining = process_mining

    @property
    def is_loaded(self) -> bool:
        return self._process_mining is not None

    def set_process_mining(self, process_mining: ProcessMining) -> None:
        if process_mining is None:
            raise ValueError("process_mining cannot be None")
        self._process_mining = process_mining

    def create(
        self,
        algorithm: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        noise_threshold: float = 0.0,
        df: Optional[pd.DataFrame] = None,
        event_log: Any = None,
    ) -> ProcessMining:
        self._process_mining = ProcessMining(
            algorithm=algorithm,
            parameters=parameters or {},
            noise_threshold=noise_threshold,
            df=df if df is not None else pd.DataFrame(),
            event_log=event_log,
        )
        return self._process_mining

    def update(self, **kwargs: Any) -> ProcessMining:
        config = self._ensure_loaded()
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config

    def get_process_mining(self) -> ProcessMining:
        return self._ensure_loaded()

    def clear(self) -> None:
        self._process_mining = None

    def _ensure_loaded(self) -> ProcessMining:
        if self._process_mining is None:
            raise ProcessMiningNotLoadedError("ProcessMining has not been set")
        return self._process_mining
