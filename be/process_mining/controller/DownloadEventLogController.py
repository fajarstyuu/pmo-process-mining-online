from pathlib import Path

from django.utils import timezone

from process_mining.model.EventLog import EventLog
from process_mining.service.EventLogService import EventLogService
from process_mining.service.DownloadEventLogService import DownloadEventLogService


class DownloadEventLogController:
	def apply(self, session_id: str | None = None):
		if not session_id:
			raise ValueError("No session ID provided")

		event_log_service = EventLogService()
		stored_event_log = EventLog()
		event_log_service.set_event_log(event_log=stored_event_log)
		event_log_service.set_id(str(session_id))

		source_path = Path(event_log_service.get_event_logs(filtered=True))
		print("Event log path: ", source_path)
		source_ext = source_path.suffix.lower()
		source_type = "csv" if source_ext == ".csv" else "xes"

		output_dir = source_path.parent
		output_dir.mkdir(parents=True, exist_ok=True)

		timestamp = timezone.now().strftime("%Y%m%d%H%M%S")
		target_ext = ".xes.gz"
		output_filename = f"{session_id}_event_log_{timestamp}{target_ext}"
		output_path = output_dir / output_filename

		download_service = DownloadEventLogService()
		_, generated_path = download_service.apply(
			file_path=str(source_path),
			output_path=str(output_path),
			file_type=source_type,
		)

		final_path = Path(generated_path) if generated_path else output_path

		return {
			"filename": final_path.name,
			"absolute_path": str(final_path.resolve()),
			"size_bytes": final_path.stat().st_size if final_path.exists() else 0,
			"source_path": str(source_path),
		}
