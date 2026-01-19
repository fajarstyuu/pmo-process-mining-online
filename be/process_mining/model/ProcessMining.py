from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict
import pandas as pd
from process_mining.model.EventLog import EventLog

@dataclass
class ProcessMining:
    algorithm: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    noise_threshold: float = 0.0
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    event_log: Optional[EventLog] = None

    # --- Getters ---
    def get_algorithm(self) -> str:
        return self.algorithm

    def get_parameters(self) -> Dict[str, Any]:
        return self.parameters

    def get_noise_threshold(self) -> float:
        return self.noise_threshold

    def get_df(self) -> pd.DataFrame:
        return self.df

    def get_event_log(self) -> EventLog:
        return self.event_log

    # --- Setters ---
    def set_algorithm(self, algorithm: str):
        self.algorithm = algorithm

    def set_parameters(self, parameters: Dict[str, Any]):
        self.parameters = parameters

    def set_noise_threshold(self, noise_threshold: float):
        self.noise_threshold = noise_threshold

    def set_df(self, df: pd.DataFrame):
        self.df = df

    def set_event_log(self, event_log: EventLog):
        self.event_log = event_log

    # --- Bulk helpers ---
    def set_all(self,
                algorithm: str,
                parameters: Dict[str, Any],
                noise_threshold: float,
                df: pd.DataFrame,
                event_log: EventLog):
        self.algorithm = algorithm
        self.parameters = parameters
        self.noise_threshold = noise_threshold
        self.df = df
        self.event_log = event_log

    def get_all(self) -> Dict[str, Any]:
        return {
            'algorithm': self.algorithm,
            'parameters': self.parameters,
            'noise_threshold': self.noise_threshold,
            'df': self.df,
            'event_log': self.event_log,
        }

