import gzip
import os

import pm4py

from process_mining.domain import LogParser
from process_mining.service.FilterService import VariantCoverageFilter, EventCoverageFilter, CaseDurationFilter, CaseSizeFilter
from process_mining.model.EventLog import EventLog
from process_mining.service.EventLogService import EventLogService

class FilterController():
    def apply(self, session_id=None, variants_coverage=1.0, events_coverage=1.0, min_case_size=None, max_case_size=None, case_duration_min=None, case_duration_max=None):
        if not session_id:
            raise ValueError("No session ID provided")

        event_log_service = EventLogService()
        event_log = EventLog()
        event_log_service.set_event_log(event_log=event_log)
        event_log_service.set_id(session_id)

        log_path = event_log_service.get_event_logs(filtered=True)
        original_ext = ".xes.gz" if log_path.lower().endswith(".xes.gz") else os.path.splitext(log_path)[1]
        parser = LogParser()
        with open(log_path, "rb") as stored_file:
            event_log = parser.parse(stored_file)

        print(f"Applying filters to event log for session_id={session_id} and named {log_path}")
        
        # #############
        # Apply Filters
        # #############
        variants_coverage_filter = VariantCoverageFilter(variant_coverage=variants_coverage)
        event_log = variants_coverage_filter.apply(event_log)

        event_coverage_filter = EventCoverageFilter(event_coverage=events_coverage)
        event_log = event_coverage_filter.apply(event_log)

        if case_duration_min is not None and case_duration_max is not None:
            performance_case_duration_filter = CaseDurationFilter(start_time_performance=case_duration_min, end_time_performance=case_duration_max)
            event_log = performance_case_duration_filter.apply(event_log)

        if min_case_size is not None and max_case_size is not None:
            performance_case_size_filter = CaseSizeFilter(min_size=min_case_size, max_size=max_case_size)
            event_log = performance_case_size_filter.apply(event_log)

        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage")
        file_path_prev = os.path.join(base_dir, session_id)
        os.makedirs(file_path_prev, exist_ok=True)

        new_filename = f"{session_id}_filter{original_ext}"
        filtered_path = os.path.join(file_path_prev, new_filename)

        if original_ext.lower() == ".csv":
            event_log.df.to_csv(filtered_path, index=False)
        else:
            log_to_save = event_log.log
            if log_to_save is None:
                log_to_save = pm4py.convert_to_event_log(event_log.df)
            if original_ext.lower() in {".xes", ".xes.gz"}:
                if original_ext.lower() == ".xes.gz":
                    temp_xes_path = filtered_path[:-3] if filtered_path.endswith(".gz") else filtered_path + ".tmp.xes"
                    pm4py.write_xes(log_to_save, temp_xes_path)
                    with open(temp_xes_path, "rb") as xes_file:
                        data = xes_file.read()
                    import gzip
                    with gzip.open(filtered_path, "wb") as gz_file:
                        gz_file.write(data)
                    os.remove(temp_xes_path)
                else:
                    pm4py.write_xes(log_to_save, filtered_path)
            else:
                event_log.df.to_csv(filtered_path, index=False)

        print(f"Filtered event log saved to {filtered_path}")

        # #################
        # Return the result
        # ################
        return None
        
