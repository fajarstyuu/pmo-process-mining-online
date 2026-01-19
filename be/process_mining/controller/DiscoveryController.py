from process_mining.domain import LogParser
from process_mining.model.FilterConfiguration import FilterConfiguration
from process_mining.service.FilterService import VariantCoverageFilter, EventCoverageFilter, CaseDurationFilter, CaseSizeFilter
from process_mining.model.ProcessMining import ProcessMining
from process_mining.service.DiscoveryService import InductiveMiner, AlphaMiner, HeuristicMiner
from process_mining.service.ProcessModelService import ProcessModelService
from process_mining.service.DirectlyFollowsGraphService import DirectlyFollowsGraphService
from process_mining.model.Statistic import Statistic
from process_mining.service.StatisticService import StatisticService
from process_mining.service.DirectlyFollowsGraphModelService import DirectlyFollowsGraphModelService
from process_mining.model.DirectlyFollowsGraph import DirectlyFollowsGraph
from process_mining.model.EventLog import EventLog
from process_mining.service.EventLogService import EventLogService
from process_mining.service.StatisticModelService import StatisticModelService
from process_mining.model.ProcessModel import ProcessModel

class DiscoveryController():
    def apply(self,uploaded=None, noise_threshold=0.0, model_name='inductive', variants_coverage=1.0, events_coverage=1.0, min_case_size=None, max_case_size=None, case_duration_min=None, case_duration_max=None):
        if not uploaded:
            raise ValueError("No file provided")

        try:
            noise_threshold = float(noise_threshold)
        except Exception:
            noise_threshold = 0.0

        parser = LogParser()
        event_log = parser.parse(uploaded)
        
        # parser = LogParser()
        # event_logs = parser.parse(uploaded)
        # event_log = EventLog()
        # event_log_service = EventLogService(event_log=event_log)
        # event_log_service.set_event_log(event_log=event_logs)


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

        
        # #######################
        # Process Model Discovery
        # #######################
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
        
        # ######################
        # Directly Follows Graph
        # ######################
        dfg_model = DirectlyFollowsGraph()
        directly_follows_graph_service = DirectlyFollowsGraphService()
        directly_follows_graph_model_service = DirectlyFollowsGraphModelService(dfg_model)
        directly_follows_graph_engine = directly_follows_graph_service.discover(event_log=event_log)
        directly_follows_graph_model_service.set_data(
            nodes=directly_follows_graph_engine[1],
            edges=directly_follows_graph_engine[0]
        )

        # ################
        # Model Statistics
        # ################
        model_statistic = Statistic()
        model_statistic_service = StatisticService()
        model_statistic_model_service = StatisticModelService(statistic=model_statistic)

        model_statistic_event = model_statistic_service.compute_events_statistics(event_log.df)
        model_statistic_case = model_statistic_service.compute_case_statistics(event_log.df)
        model_statistic_variants = model_statistic_service.compute_variant_statistics(event_log.df)
        model_statistic_resources = model_statistic_service.compute_resource_statistics(event_log.df)
        
        model_statistic_model_service.set_statistic_data(
            events=model_statistic_event,
            case=model_statistic_case,
            variants=model_statistic_variants,
            resources=model_statistic_resources
        )

        # #################
        # Return the result
        # ################
        return {
            "petri_net": model_process_service.to_cytoscape(),
            "dfg": {
                "nodes": directly_follows_graph_model_service.get_nodes(),
                "edges": directly_follows_graph_model_service.get_edges()
            },
            "model_statistics": model_statistic_model_service.get_statistic_data()
        }
