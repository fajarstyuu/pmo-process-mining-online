from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple
import pm4py
import os
import pandas as pd
from .domain import EventLog, ProcessModel


class ProcessDiscoveryEngine(ABC):
    @abstractmethod
    def discover(self, event_log: EventLog, noise_threshold: float = 0.0) -> ProcessModel:
        raise NotImplementedError()


class InductiveMiner(ProcessDiscoveryEngine):
    def discover(self, event_log: EventLog, noise_threshold: float = 0.0) -> ProcessModel:
        # use pm4py's inductive miner
        net, im, fm = pm4py.discover_petri_net_inductive(event_log.df, noise_threshold=noise_threshold)
        dfg_metrics = _compute_dfg_metrics(event_log.df)
        return _build_process_model_from_petri(net, im, fm, noise_threshold, dfg_metrics=dfg_metrics)


class AlphaMiner(ProcessDiscoveryEngine):
    def discover(self, event_log: EventLog, noise_threshold: float = 0.0) -> ProcessModel:
        net, im, fm = pm4py.discover_petri_net_alpha(event_log.df)
        dfg_metrics = _compute_dfg_metrics(event_log.df)
        return _build_process_model_from_petri(net, im, fm, noise_threshold, dfg_metrics=dfg_metrics)


class HeuristicMiner(ProcessDiscoveryEngine):
    def discover(self, event_log: EventLog, noise_threshold: float = 0.0) -> ProcessModel:
        # heuristic miner in pm4py needs a heuristic net step
        net, im, fm = pm4py.discover_petri_net_heuristics(event_log.df)
        dfg_metrics = _compute_dfg_metrics(event_log.df)
        return _build_process_model_from_petri(net, im, fm, noise_threshold, dfg_metrics=dfg_metrics)


def _build_process_model_from_petri(petri_net, initial_marking, final_marking, noise_threshold: float = 0.0, dfg_metrics: Dict[Tuple[str,str], Dict] = None) -> ProcessModel:
    # Build nodes and edges for Cytoscape-style frontend
    nodes = []
    edges = []

    # maps
    place_id_map = {}
    trans_id_map = {}

    # places
    for p in petri_net.places:
        pid = f"p_{getattr(p, 'name', str(id(p)))}"
        place_id_map[getattr(p, 'name', str(id(p)))] = pid
        nodes.append({
            'data': {
                'id': pid,
                'label': getattr(p, 'name', str(p)),
                'type': 'place'
            }
        })

    # transitions
    for t in petri_net.transitions:
        tid = f"t_{getattr(t, 'name', str(id(t)))}"
        trans_id_map[getattr(t, 'name', str(id(t)))] = tid
        label = getattr(t, 'label', None) if hasattr(t, 'label') else None
        visible = bool(label)
        nodes.append({
            'data': {
                'id': tid,
                'label': label if label else getattr(t, 'name', str(t)),
                'type': 'transition',
                'visible': visible
            }
        })

    # arcs
    for idx, arc in enumerate(petri_net.arcs):
        src_name = getattr(arc.source, 'name', None) or str(id(arc.source))
        tgt_name = getattr(arc.target, 'name', None) or str(id(arc.target))

        src_id = place_id_map.get(src_name) or trans_id_map.get(src_name) or f"unknown_{src_name}"
        tgt_id = place_id_map.get(tgt_name) or trans_id_map.get(tgt_name) or f"unknown_{tgt_name}"

        edges.append({
            'data': {
                'id': f"e_{idx}",
                'source': src_id,
                'target': tgt_id
            }
        })

    model_stats = {
        'number_of_places': len(petri_net.places),
        'number_of_transitions': len(petri_net.transitions),
        'number_of_arcs': len(petri_net.arcs),
        'noise_threshold_used': noise_threshold
    }

    # If DFG metrics are provided, create DFG edges between transitions (activity->activity)
    if dfg_metrics:
        # map activity label -> transition id (use visible transitions)
        label_to_tid = {}
        for t in petri_net.transitions:
            label = getattr(t, 'label', None)
            tid = trans_id_map.get(getattr(t, 'name', str(id(t))))
            if label and tid:
                label_to_tid[label] = tid

        dfg_idx = 0
        for (a, b), stats in dfg_metrics.items():
            src_tid = label_to_tid.get(a)
            tgt_tid = label_to_tid.get(b)
            if src_tid and tgt_tid:
                edges.append({
                    'data': {
                        'id': f"dfg_{dfg_idx}",
                        'source': src_tid,
                        'target': tgt_tid,
                        'frequency': int(stats.get('count', 0)),
                        'performance_seconds': float(stats.get('avg_time', 0.0))
                    },
                    'type': 'dfg'
                })
                dfg_idx += 1

    return ProcessModel(nodes=nodes, edges=edges, model_statistics=model_stats)


def _compute_dfg_metrics(df: pd.DataFrame) -> Dict[Tuple[str, str], Dict]:
    """
    Compute directly-follows metrics from a dataframe with columns
    'case:concept:name', 'concept:name', and optionally 'time:timestamp'.

    Returns mapping (a,b) -> {'count': int, 'avg_time': seconds_float}
    """
    dfg = {}
    has_time = 'time:timestamp' in df.columns

    # Ensure timestamp column is datetime if present
    if has_time:
        try:
            df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], errors='coerce')
        except Exception:
            has_time = False

    # group by case
    grouped = df.groupby('case:concept:name')
    for case_id, group in grouped:
        # sort by timestamp if available otherwise by index
        if has_time and group['time:timestamp'].notnull().any():
            group = group.sort_values('time:timestamp')
        else:
            group = group.sort_index()

        activities = list(group['concept:name'])
        times = list(group['time:timestamp']) if has_time else [None] * len(activities)

        for i in range(len(activities) - 1):
            a = activities[i]
            b = activities[i + 1]
            key = (a, b)
            delta = None
            if has_time and times[i] is not None and times[i+1] is not None:
                try:
                    delta = (times[i+1] - times[i]).total_seconds()
                except Exception:
                    delta = None

            entry = dfg.get(key)
            if not entry:
                entry = {'count': 0, 'total_time': 0.0, 'time_samples': 0}
                dfg[key] = entry

            entry['count'] += 1
            if delta is not None:
                entry['total_time'] += float(delta)
                entry['time_samples'] += 1

    # finalize averages
    result = {}
    for k, v in dfg.items():
        avg = (v['total_time'] / v['time_samples']) if v['time_samples'] > 0 else 0.0
        result[k] = {'count': v['count'], 'avg_time': avg}

    return result
