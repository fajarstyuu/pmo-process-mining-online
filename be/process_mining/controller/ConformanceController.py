from process_mining.domain import LogParser
from process_mining.model.FilterConfiguration import FilterConfiguration
from process_mining.service.EventLogService import EventLogService
from process_mining.model.EventLog import EventLog
from process_mining.service.FilterService import VariantCoverageFilter, EventCoverageFilter, CaseDurationFilter, CaseSizeFilter
from process_mining.model.ProcessMining import ProcessMining
from process_mining.service.DiscoveryService import InductiveMiner, AlphaMiner, HeuristicMiner
from process_mining.service.ProcessModelService import ProcessModelService
from process_mining.service.ConformanceService import ConformanceService
from process_mining.model.ProcessModel import ProcessModel

class ConformanceController():
    def apply(self, session_id=None, noise_threshold=0.0, model_name='inductive', variants_coverage=1.0, events_coverage=1.0, min_case_size=None, max_case_size=None, case_duration_min=None, case_duration_max=None, use_filtered_log=False):
        if not session_id:
            raise ValueError("No session ID provided")

        try:
            noise_threshold = float(noise_threshold)
        except Exception:
            noise_threshold = 0.0

        event_log_service = EventLogService()
        stored_event_log = EventLog()
        event_log_service.set_event_log(event_log=stored_event_log)
        event_log_service.set_id(str(session_id))

        log_path = event_log_service.get_event_logs(filtered=use_filtered_log)
        parser = LogParser()
        with open(log_path, "rb") as stored_file:
            event_log_origin = parser.parse(stored_file)

        print(f"Computing conformance for event log for session_id={session_id} using model={model_name} with noise_threshold={noise_threshold} on log named {log_path}")

        filtering_config = FilterConfiguration()
        filtering_config.set_all(
            variant_coverage=variants_coverage,
            event_coverage=events_coverage,
            start_time_performance=case_duration_min,
            end_time_performance=case_duration_max,
            max_size_performance=max_case_size,
            min_size_performance=min_case_size,
            event_log=event_log_origin
        )

        variants_coverage_filter = VariantCoverageFilter(variant_coverage=filtering_config.get_variant_coverage())
        event_log = variants_coverage_filter.apply(event_log_origin)

        event_coverage_filter = EventCoverageFilter(event_coverage=filtering_config.get_event_coverage())
        event_log = event_coverage_filter.apply(event_log)

        if case_duration_min is not None and case_duration_max is not None:
            performance_case_duration_filter = CaseDurationFilter(start_time_performance=filtering_config.get_start_time_performance(), end_time_performance=filtering_config.get_end_time_performance())
            event_log = performance_case_duration_filter.apply(event_log)

        if min_case_size is not None and max_case_size is not None:
            performance_case_size_filter = CaseSizeFilter(min_size=filtering_config.get_min_size_performance(), max_size=filtering_config.get_max_size_performance())
            event_log = performance_case_size_filter.apply(event_log)

        model_key = str(model_name).lower() if model_name else 'inductive'
        process_miner = ProcessMining(event_log=event_log, noise_threshold=noise_threshold)
        if model_key == 'alpha':
            process_miner.set_algorithm('alpha')
        elif model_key == 'heuristic':
            process_miner.set_algorithm('heuristic')
        else:
            process_miner.set_algorithm('inductive')

        engine = process_miner.get_algorithm()
        if engine == 'alpha':
            net, im, fm = AlphaMiner().discover(event_log, noise_threshold)
        elif engine == 'heuristic':
            net, im, fm = HeuristicMiner().discover(event_log, noise_threshold)
        else:
            net, im, fm = InductiveMiner().discover(event_log, noise_threshold)

        model_process = ProcessModel()
        model_process_service = ProcessModelService(model_process)

        conformance_metrics = ConformanceService.apply(event_log_origin, net, im, fm)
        model_process_service.set_evaluation_metrics(conformance_metrics)
        return model_process_service.get_evaluation_metrics()
