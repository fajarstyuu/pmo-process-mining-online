
from process_mining.utils.converter import Converter

class DownloadEventLogService:
    def apply(self, file_path: str, output_path: str, file_type: str = 'xes'):
        if file_type == 'csv':
            event_log_data, output_paths = Converter.csv_to_log(csv_path=file_path, output_path=output_path)
        else:
            event_log_data, output_paths = Converter.xes_to_log(xes_path=file_path, output_path=output_path)
        return event_log_data, output_paths