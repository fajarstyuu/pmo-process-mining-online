from typing import Any, Dict, Optional, Tuple

import pandas as pd
import os
from process_mining.model.EventLog import EventLog


class EventLogNotLoadedError(RuntimeError):
	"""Raised when the stored EventLog is requested before being set."""


class EventLogService:
	"""Minimal read/write helper for EventLog objects."""

	def __init__(self, event_log: Optional[EventLog] = None) -> None:
		self._event_log = event_log

	@property
	def is_loaded(self) -> bool:
		return self._event_log is not None
	
	def validate_loaded(self, event_log) -> None:
		if event_log is None:
			raise EventLogNotLoadedError("EventLog log is not loaded.")
		filename = event_log.name.lower() or event_log.filename.lower()
		if not (filename.endswith('.xes') or filename.endswith('.csv') or filename.endswith('.xes.gz')):
			raise EventLogNotLoadedError("EventLog log is not in a valid format. Only .xes and .csv are supported.")
	
	def set_id(self, id: str) -> None:
		self._event_log.set_id(id)

	def set_event_log(self, event_log: EventLog) -> None:
		if event_log is None:
			raise ValueError("event_log cannot be None")
		self._event_log = event_log

	def set_from_dataframe(self, df: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None, log: Any = None) -> None:
		if not isinstance(df, pd.DataFrame):
			raise TypeError("df must be a pandas DataFrame")
		self._event_log = EventLog(df=df, metadata=metadata or {}, log=log)

	def update_metadata(self, metadata: Dict[str, Any]) -> None:
		log = self._ensure_loaded()
		log.metadata.update(metadata)

	def clear(self) -> None:
		self._event_log = None

	def get_id(self) -> str:
		return self._event_log.get_id()

	def get_event_logs(self, filtered: bool = False) -> EventLog:
		event_log_id = self.get_id()
		base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage")
		path = os.path.join(base_dir, event_log_id)
		if not os.path.exists(path):
			raise FileNotFoundError("Event log directory not found")
		files = [
    	    f.path for f in os.scandir(path)
    	    if f.is_file()
    	]
		print(f"Available event log files for session_id={event_log_id}: {files}")
		if not files:
			raise FileNotFoundError("No event log file found")

		if filtered:
			filtered_files = [path for path in files if "_filter" in os.path.basename(path)]
			if filtered_files:
				return max(filtered_files, key=os.path.getmtime)
			# fall back to whatever exists
			return max(files, key=os.path.getmtime)
		else:
			non_filtered_files = [path for path in files if "_filter" not in os.path.basename(path)]
			if non_filtered_files:
				return max(non_filtered_files, key=os.path.getmtime)
			# fall back to available filtered version
			return max(files, key=os.path.getmtime)

	def get_event_log(self) -> EventLog:
		return self._ensure_loaded()
