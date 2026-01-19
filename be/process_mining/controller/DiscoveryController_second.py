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
    def apply(self, session_id=None, model_name='inductive', noise_threshold=0.0):
        if not session_id:
            raise ValueError("No session ID provided")
        try:
            noise_threshold = float(noise_threshold)
        except Exception:
            noise_threshold = 0.0

        event_log_service = EventLogService()
        event_log = EventLog()
        event_log_service.set_event_log(event_log=event_log)
        event_log_service.set_id(session_id)

        log_path = event_log_service.get_event_logs(filtered=True)
        parser = LogParser()
        with open(log_path, "rb") as stored_file:
            event_log = parser.parse(stored_file)

        print(f"Discovering process discovery for session_id={session_id} using model={model_name} with noise_threshold={noise_threshold} on log named {log_path}")
        
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
            }
        }
