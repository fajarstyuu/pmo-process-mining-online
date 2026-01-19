from dataclasses import dataclass, field
from typing import List, Dict, Any
import pandas as pd

@dataclass
class EventLog:
    id: str = None
    df: pd.DataFrame = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    log: Any = None

    def set_id(self, id: str) -> None:
        self.id = id

    def set_df(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        self.df = df

    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        self.metadata = metadata

    def set_log(self, log: Any) -> None:
        self.log = log

    def get_id(self) -> str:
        return self.id

    def get_df(self) -> pd.DataFrame:
        return self.df
    
    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata
    
    def get_log(self) -> Any:
        return self.log

