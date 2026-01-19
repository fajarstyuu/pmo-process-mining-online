import os
from datetime import datetime
from ..service.EventLogService import EventLogService
from process_mining.utils.converter import Converter

class LoadEventLogController:
    def apply(self, uploaded=None, session_id=None):
        if not uploaded:
            raise ValueError("No file provided")
        if not session_id:
            raise ValueError("No session ID provided")
        
        print(f"[{datetime.now()}] LoadEventLogController: Applying upload for session_id={session_id}")

        event_log_service = EventLogService()
        event_log_service.validate_loaded(uploaded)
        event_log_service.set_event_log(uploaded)

        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage")
        file_path_prev = os.path.join(base_dir, session_id)
        os.makedirs(file_path_prev, exist_ok=True)

        original_name = getattr(uploaded, 'name', None) or getattr(uploaded, 'filename', '')
        _, ext = os.path.splitext(original_name)
        ext = (ext or '').lower()

        if ext == '.csv':
            csv_temp = os.path.join(file_path_prev, f"{session_id}_uploaded.csv")
            uploaded.file.seek(0)
            with open(csv_temp, 'wb') as temp_file:
                temp_file.write(uploaded.file.read())

            output_path = os.path.join(file_path_prev, f"{session_id}.xes")
            Converter.csv_to_log(csv_temp, output_path=output_path)
            try:
                os.remove(csv_temp)
            except OSError:
                pass
        else:
            new_filename = f"{session_id}{ext}"
            uploaded.file.seek(0)
            with open(os.path.join(file_path_prev, new_filename), 'wb') as file:
                file.write(uploaded.file.read())

        # def apply(self, uploaded=None, session_id=None):
        # if not uploaded:
        #     raise ValueError("No file provided")
        # if not session_id:
        #     raise ValueError("No session ID provided")

        # event_log_service = EventLogService()
        # event_log_service.validate_loaded(uploaded)
        # event_log_service.set_event_log(uploaded)

        # base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage")
        # file_path_prev = os.path.join(base_dir, session_id)
        # os.makedirs(file_path_prev, exist_ok=True)

        # _, ext = os.path.splitext(uploaded.name)
        # base_name = session_id

        # # file pertama tanpa suffix
        # new_filename = f"{base_name}{ext}"
        # base_path = os.path.join(file_path_prev)

        # if os.path.exists(os.path.join(base_path, new_filename)):
        #     max_index = 0
        #     pattern = re.compile(rf"^{re.escape(base_name)}_(\d+){re.escape(ext)}$")

        #     for filename in os.listdir(base_path):
        #         match = pattern.match(filename)
        #         if match:
        #             idx = int(match.group(1))
        #             max_index = max(max_index, idx)

        #     new_filename = f"{base_name}_{max_index + 1}{ext}"
        # with open(os.path.join(file_path_prev, new_filename), 'wb') as file:
        #     file.write(uploaded.file.read())