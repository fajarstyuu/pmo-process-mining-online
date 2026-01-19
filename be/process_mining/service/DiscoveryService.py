from abc import ABC, abstractmethod
import pm4py
import pandas as pd
# from .domain import DirectlyFollowsGraph, EventLog, ProcessModel
from process_mining.model import ProcessModel, EventLog
class DiscoveryService(ABC):
    @abstractmethod
    def discover(self, event_log: EventLog, noise_threshold: float = 0.0) -> ProcessModel:
        raise NotImplementedError()

class InductiveMiner(DiscoveryService):
    def discover(self, event_log: EventLog, noise_threshold: float = 0.0):
        # use pm4py's inductive miner
        net, im, fm = pm4py.discover_petri_net_inductive(event_log.df, noise_threshold=noise_threshold)
        # pm = _build_process_model_from_petri(net, im, fm, noise_threshold, dfg_map=dfg_map, activity_freq=activity_freq, avg_time_from=avg_time_from, event_log_df=event_log.df)
        # also expose dfg list for convenience
        # pm.conformance_metrics = _compute_conformance_metrics(event_log, net, im, fm)
        return net, im, fm

class AlphaMiner(DiscoveryService):
    def discover(self, event_log: EventLog, noise_threshold: float = 0.0):
        # Alpha miner
        net, im, fm = pm4py.discover_petri_net_alpha(event_log.df)
        # pm = _build_process_model_from_petri(net, im, fm, noise_threshold, dfg_map=dfg_map, activity_freq=activity_freq, avg_time_from=avg_time_from, event_log_df=event_log.df)
        return net, im, fm


class HeuristicMiner(DiscoveryService):
    def discover(self, event_log: EventLog, noise_threshold: float = 0.0):
        # heuristic miner in pm4py needs a heuristic net step
        net, im, fm = pm4py.discover_petri_net_heuristics(event_log.df)
        # pm = _build_process_model_from_petri(net, im, fm, noise_threshold, dfg_map=dfg_map, activity_freq=activity_freq, avg_time_from=avg_time_from, event_log_df=event_log.df)
        return net, im, fm
