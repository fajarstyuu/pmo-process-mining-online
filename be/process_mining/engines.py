from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict
import pm4py
import os
import pandas as pd
from .domain import EventLog, ProcessModel
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
import json

class ProcessDiscoveryEngine(ABC):
    @abstractmethod
    def discover(self, event_log: EventLog, noise_threshold: float = 0.0) -> ProcessModel:
        raise NotImplementedError()


class InductiveMiner(ProcessDiscoveryEngine):
    def discover(self, event_log: EventLog, noise_threshold: float = 0.0) -> ProcessModel:
        # use pm4py's inductive miner
        net, im, fm = pm4py.discover_petri_net_inductive(event_log.df, noise_threshold=noise_threshold)
        # compute DFG and activity metrics
        dfg_map = _compute_dfg_map(event_log.df)
        activity_freq = _compute_activity_freq(event_log.df)
        avg_time_from = _compute_avg_time_from_activity(dfg_map)
        pm = _build_process_model_from_petri(net, im, fm, noise_threshold, dfg_map=dfg_map, activity_freq=activity_freq, avg_time_from=avg_time_from, event_log_df=event_log.df)
        # also expose dfg list for convenience
        pm.dfg = _dfg_map_to_list(dfg_map)
        pm.conformance_metrics = _compute_conformance_metrics(event_log, net, im, fm)
        pm.model_statistics["events"] = _compute_events_statistics(event_log.df)
        pm.model_statistics["case"] = _compute_case_statistics(event_log.df)
        return pm


class AlphaMiner(ProcessDiscoveryEngine):
    def discover(self, event_log: EventLog, noise_threshold: float = 0.0) -> ProcessModel:
        # Alpha miner
        net, im, fm = pm4py.discover_petri_net_alpha(event_log.df)
        dfg_map = _compute_dfg_map(event_log.df)
        activity_freq = _compute_activity_freq(event_log.df)
        avg_time_from = _compute_avg_time_from_activity(dfg_map)
        pm = _build_process_model_from_petri(net, im, fm, noise_threshold, dfg_map=dfg_map, activity_freq=activity_freq, avg_time_from=avg_time_from, event_log_df=event_log.df)
        pm.dfg = _dfg_map_to_list(dfg_map)
        pm.conformance_metrics = _compute_conformance_metrics(event_log, net, im, fm)
        return pm


class HeuristicMiner(ProcessDiscoveryEngine):
    def discover(self, event_log: EventLog, noise_threshold: float = 0.0) -> ProcessModel:
        # heuristic miner in pm4py needs a heuristic net step
        net, im, fm = pm4py.discover_petri_net_heuristics(event_log.df)
        dfg_map = _compute_dfg_map(event_log.df)
        activity_freq = _compute_activity_freq(event_log.df)
        avg_time_from = _compute_avg_time_from_activity(dfg_map)
        pm = _build_process_model_from_petri(net, im, fm, noise_threshold, dfg_map=dfg_map, activity_freq=activity_freq, avg_time_from=avg_time_from, event_log_df=event_log.df)
        pm.dfg = _dfg_map_to_list(dfg_map)
        pm.conformance_metrics = _compute_conformance_metrics(event_log, net, im, fm)
        return pm


def _build_process_model_from_petri(petri_net, initial_marking, final_marking, noise_threshold: float = 0.0, dfg_map: Dict[tuple, dict] = None, activity_freq: Dict[str, int] = None, avg_time_from: Dict[str, float] = None, event_log_df: Optional[pd.DataFrame] = None) -> ProcessModel:
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

        # determine if this arc is adjacent to a transition with a label
        src_label = getattr(arc.source, 'label', None) if hasattr(arc.source, 'label') or hasattr(arc.source, 'name') else None
        if src_label is None and hasattr(arc.source, 'label'):
            src_label = getattr(arc.source, 'label', None)
        tgt_label = getattr(arc.target, 'label', None) if hasattr(arc.target, 'label') or hasattr(arc.target, 'name') else None

        frequency = None
        performance_seconds = None

        # prefer metrics from transition side if available
        if src_label:
            frequency = (activity_freq.get(src_label) if activity_freq else None) if activity_freq else None
            performance_seconds = (avg_time_from.get(src_label) if avg_time_from else None) if avg_time_from else None
        elif tgt_label:
            frequency = (activity_freq.get(tgt_label) if activity_freq else None) if activity_freq else None
            performance_seconds = (avg_time_from.get(tgt_label) if avg_time_from else None) if avg_time_from else None

        data = {
            'id': f"e_{idx}",
            'source': src_id,
            'target': tgt_id
        }
        if frequency is not None:
            try:
                data['frequency'] = int(frequency)
            except Exception:
                data['frequency'] = frequency
        if performance_seconds is not None:
            try:
                data['performance_seconds'] = float(performance_seconds)
            except Exception:
                data['performance_seconds'] = performance_seconds

        edges.append({
            'data': data
        })

    model_stats = {
        'number_of_places': len(petri_net.places),
        'number_of_transitions': len(petri_net.transitions),
        'number_of_arcs': len(petri_net.arcs),
        'noise_threshold_used': noise_threshold
    }

    # compute number_of_cases and number_of_variants when event log data is provided
    number_of_cases = 0
    number_of_variants = 0
    if event_log_df is not None and 'case:concept:name' in event_log_df.columns and 'concept:name' in event_log_df.columns:
        try:
            # operate on a copy to avoid mutating the original df
            el_df = event_log_df.copy()
            has_time = 'time:timestamp' in el_df.columns
            if has_time:
                try:
                    el_df['time:timestamp'] = pd.to_datetime(el_df['time:timestamp'], errors='coerce')
                except Exception:
                    has_time = False

            grouped = el_df.groupby('case:concept:name')
            number_of_cases = len(grouped)
            variants = set()
            for case_id, group in grouped:
                if has_time and group['time:timestamp'].notnull().any():
                    group = group.sort_values('time:timestamp')
                else:
                    group = group.sort_index()
                seq = tuple(group['concept:name'].tolist())
                variants.add(seq)
            number_of_variants = len(variants)
        except Exception:
            # defensive: fall back to zeros if anything goes wrong
            number_of_cases = 0
            number_of_variants = 0

    model_stats['number_of_cases'] = int(number_of_cases)
    model_stats['number_of_variants'] = int(number_of_variants)

    return ProcessModel(nodes=nodes, edges=edges, model_statistics=model_stats)


def _compute_dfg_map(df: pd.DataFrame) -> Dict[tuple, dict]:
    """
    Compute directly-follows metrics and return mapping (a,b) -> {'count':int,'avg_time':float}
    """
    dfg = {}
    if df is None or 'case:concept:name' not in df.columns or 'concept:name' not in df.columns:
        return dfg

    has_time = 'time:timestamp' in df.columns
    if has_time:
        try:
            df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], errors='coerce')
        except Exception:
            has_time = False

    grouped = df.groupby('case:concept:name')
    for case_id, group in grouped:
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
            if has_time and times[i] is not None and times[i + 1] is not None:
                try:
                    delta = (times[i + 1] - times[i]).total_seconds()
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

    # finalize
    result = {}
    for k, v in dfg.items():
        avg = (v['total_time'] / v['time_samples']) if v['time_samples'] > 0 else 0.0
        result[k] = {'count': v['count'], 'avg_time': avg}

    return result


def _compute_activity_freq(df: pd.DataFrame) -> Dict[str, int]:
    if df is None or 'concept:name' not in df.columns:
        return {}
    try:
        return df['concept:name'].value_counts().to_dict()
    except Exception:
        return {}


def _compute_avg_time_from_activity(dfg_map: Dict[tuple, dict]) -> Dict[str, float]:
    """Compute weighted average time from activity a to its successors."""
    res = {}
    for (a, b), stats in dfg_map.items():
        cnt = stats.get('count', 0)
        avg = stats.get('avg_time', 0.0)
        if a not in res:
            res[a] = {'total_time': 0.0, 'count': 0}
        res[a]['total_time'] += avg * cnt
        res[a]['count'] += cnt

    out = {}
    for a, v in res.items():
        out[a] = (v['total_time'] / v['count']) if v['count'] > 0 else 0.0
    return out


def _dfg_map_to_list(dfg_map: Dict[tuple, dict]) -> List[Dict[str, object]]:
    out = []
    for (a, b), stats in dfg_map.items():
        out.append({'source': a, 'target': b, 'frequency': int(stats.get('count', 0)), 'performance_seconds': float(stats.get('avg_time', 0.0))})
    return out


def _compute_dfg_list(df: pd.DataFrame) -> List[Dict[str, object]]:
    """
    Compute directly-follows metrics and return as a list of dicts:
    [{ source: str, target: str, frequency: int, performance_seconds: float }, ...]
    """
    result = []
    # quick defensive checks
    if df is None or 'case:concept:name' not in df.columns or 'concept:name' not in df.columns:
        return result

    has_time = 'time:timestamp' in df.columns
    if has_time:
        try:
            df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], errors='coerce')
        except Exception:
            has_time = False

    grouped = df.groupby('case:concept:name')
    dfg = {}
    for case_id, group in grouped:
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
            if has_time and times[i] is not None and times[i + 1] is not None:
                try:
                    delta = (times[i + 1] - times[i]).total_seconds()
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

    for (a, b), v in dfg.items():
        avg = (v['total_time'] / v['time_samples']) if v['time_samples'] > 0 else 0.0
        result.append({'source': a, 'target': b, 'frequency': int(v['count']), 'performance_seconds': float(avg)})

    return result

def _compute_conformance_metrics(event_log: EventLog, petri_net, initial_marking, final_marking) -> Dict[str, float]:
    """
    Compute conformance metrics: fitness, precision, generalization, simplicity
    """
    metrics = {}

    # Fitness
    fitness = fitness_evaluator.apply(event_log.df, petri_net, initial_marking, final_marking)
    metrics['fitness'] = fitness

    # Precision
    precision = precision_evaluator.apply(event_log.df, petri_net, initial_marking, final_marking)
    metrics['precision'] = precision

    # Generalization
    generalization = generalization_evaluator.apply(event_log.df, petri_net, initial_marking, final_marking)
    metrics['generalization'] = generalization

    # Simplicity
    simplicity = simplicity_evaluator.apply(petri_net)
    metrics['simplicity'] = simplicity

    return metrics

def _compute_events_statistics(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Compute event statistics: frequency of each activity
    """
    """
    Return a list of activity statistics. Each item contains:
    { activity, count, percent, median, mean, range }

    Notes:
    - This function does NOT mutate the input DataFrame.
    - If timestamps or case ids are missing, duration fields will be None but
      activity counts/percent will still be returned.
    - Exceptions are not silently swallowed; errors will propagate to the caller.
    """
    if df is None or 'concept:name' not in df.columns:
        return []

    # work on a copy to avoid mutating caller's dataframe
    df2 = df.copy()

    try:
        # basic counts / percent (always available if 'concept:name' exists)
        event_counts = df2['concept:name'].value_counts()
        total = int(event_counts.sum()) if event_counts.sum() is not None else 0
        event_percentage = (event_counts / total * 100).round(2) if total > 0 else event_counts * 0.0

        # prepare duration statistics only if both case id and timestamp are present
        has_case = 'case:concept:name' in df2.columns
        has_time = has_case and ('time:timestamp' in df2.columns)

        duration_stats = None
        if has_time:
            df2['time:timestamp'] = pd.to_datetime(df2['time:timestamp'], errors='coerce')
            # duration as difference with previous event within the same case
            df2['duration'] = (df2['time:timestamp'] - df2.groupby('case:concept:name')['time:timestamp'].shift(1)).dt.total_seconds()
            duration_stats = df2.groupby('concept:name')['duration'].agg(['mean', 'median', 'min', 'max'])
            duration_stats['range'] = duration_stats['max'] - duration_stats['min']
            duration_stats = duration_stats.round(2)

        result: List[Dict[str, Any]] = []
        for activity in event_counts.index:
            count = int(event_counts[activity])
            percent = float(event_percentage[activity]) if activity in event_percentage.index else 0.0

            mean_val = None
            median_val = None
            range_val = None
            if duration_stats is not None and activity in duration_stats.index:
                row = duration_stats.loc[activity]
                # convert possible NaN -> None, numpy types -> native python types
                def _as_float_or_none(x):
                    return None if pd.isna(x) else float(x)

                mean_val = _as_float_or_none(row.get('mean'))
                median_val = _as_float_or_none(row.get('median'))
                range_val = _as_float_or_none(row.get('range'))

            result.append({
                'activity': activity,
                'count': count,
                'percent': percent,
                'median': median_val,
                'mean': mean_val,
                'range': range_val
            })

        return result

    except Exception:
        # do not silently swallow errors; let caller decide how to handle them
        raise

def _compute_case_statistics(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Compute case statistics: frequency of each case
    """
    """
        Return a list of case statistics. Each item contains keys like:
        {
            "case:concept:name": <case id>,
            "start_time": "YYYY-MM-DD HH:MM:SS.ffffff" or None,
            "end_time": "YYYY-MM-DD HH:MM:SS.ffffff" or None,
            "event_count": int,
            "duration": float (seconds, rounded to 2 decimals) or None
        }

    Notes:
    - This function does NOT mutate the input DataFrame.
    - If case ids are missing, an empty list is returned.
    - If timestamps are missing, start_time/end_time/duration_seconds will be None.
    - Exceptions are not silently swallowed; errors will propagate to the caller.
    """
    if df is None or 'case:concept:name' not in df.columns:
        return []

    # operate on a copy to avoid mutating caller dataframe
    df2 = df.copy()

    # detect availability of timestamp
    has_time = 'time:timestamp' in df2.columns
    if has_time:
        df2['time:timestamp'] = pd.to_datetime(df2['time:timestamp'], errors='coerce')

    # basic counts
    case_counts = df2['case:concept:name'].value_counts()

    # pre-compute per-case start/end times if time is available
    time_bounds = None
    if has_time:
        # groupby min/max; NaT will remain and handled later
        time_bounds = df2.groupby('case:concept:name')['time:timestamp'].agg(['min', 'max'])

    result: List[Dict[str, Any]] = []
    for case_id in case_counts.index:
        event_count = int(case_counts[case_id])

        start_str = None
        end_str = None
        duration_val = None

        if time_bounds is not None and case_id in time_bounds.index:
            start_ts = time_bounds.loc[case_id, 'min']
            end_ts = time_bounds.loc[case_id, 'max']
            if not pd.isna(start_ts):
                try:
                    start_str = pd.Timestamp(start_ts).strftime("%Y-%m-%d %H:%M:%S.%f")
                except Exception:
                    start_str = None
            if not pd.isna(end_ts):
                try:
                    end_str = pd.Timestamp(end_ts).strftime("%Y-%m-%d %H:%M:%S.%f")
                except Exception:
                    end_str = None
            if (not pd.isna(start_ts)) and (not pd.isna(end_ts)):
                try:
                    duration_val = round((pd.Timestamp(end_ts) - pd.Timestamp(start_ts)).total_seconds(), 2)
                except Exception:
                    duration_val = None

        result.append({
            'case_id': case_id,
            'start_time': start_str,
            'end_time': end_str,
            'event_count': event_count,
            'duration': duration_val
        })

    return result