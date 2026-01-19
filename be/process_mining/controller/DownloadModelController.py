from process_mining.domain import LogParser
from process_mining.model.EventLog import EventLog
from process_mining.service.EventLogService import EventLogService
from process_mining.service.FilterService import VariantCoverageFilter, EventCoverageFilter, CaseDurationFilter, CaseSizeFilter
from process_mining.service.DiscoveryService import InductiveMiner, AlphaMiner, HeuristicMiner
from process_mining.service.ProcessModelService import ProcessModelService
from process_mining.model.ProcessModel import ProcessModel
from process_mining.service.DownloadModelService import DownloadModelService

class DownloadModelController:
    def apply(self, session_id=None, noise_threshold=0.0, model_name='inductive', variants_coverage=1.0, events_coverage=1.0, min_case_size=None, max_case_size=None, case_duration_min=None, case_duration_max=None, use_filtered_log=True):
        if not session_id:
            raise ValueError("Session ID must be provided")

        try:
            noise_threshold = float(noise_threshold)
        except Exception:
            noise_threshold = 0.0

        parser = LogParser()
        event_log_service = EventLogService()
        stored_event_log = EventLog()
        event_log_service.set_event_log(event_log=stored_event_log)
        event_log_service.set_id(str(session_id))
        log_path = event_log_service.get_event_logs(filtered=use_filtered_log)
        with open(log_path, "rb") as stored_file:
            event_log = parser.parse(stored_file)

        print(f"DownloadModelController: Loaded event log from session_id={session_id} using file {log_path}")

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

        model_key = str(model_name).lower() if model_name else 'inductive'
        if model_key == 'alpha':
            net, im, fm = AlphaMiner().discover(event_log, noise_threshold)
        elif model_key == 'heuristic':
            net, im, fm = HeuristicMiner().discover(event_log, noise_threshold)
        else:
            net, im, fm = InductiveMiner().discover(event_log, noise_threshold)

        model_process = ProcessModel()
        model_process_service = ProcessModelService(model_process)
        model_process_service.build_process_model_from_petri(
            net,
            im,
            fm,
            noise_threshold,
            event_log_df=event_log.df
        )

        download_service = DownloadModelService()
        pnml_file = download_service.apply(net, im, fm)

        return pnml_file