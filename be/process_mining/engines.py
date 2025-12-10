from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict
import math
import pm4py
import os
import pandas as pd
from .domain import DirectlyFollowsGraph, EventLog, ProcessModel
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

from pm4py.algo.evaluation.replay_fitness.variants import token_replay as fitness_evaluator
from pm4py.algo.evaluation.precision.variants import etconformance_token as precision_evaluator
from pm4py.algo.evaluation.precision.variants import etconformance_token as generalization_evaluator

# from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_evaluator
# from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
# from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery

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
        pm.conformance_metrics = _compute_conformance_metrics(event_log, net, im, fm)
        return pm

class AlphaMiner(ProcessDiscoveryEngine):
    def discover(self, event_log: EventLog, noise_threshold: float = 0.0) -> ProcessModel:
        # Alpha miner
        net, im, fm = pm4py.discover_petri_net_alpha(event_log.df)
        dfg_map = _compute_dfg_map(event_log.df)
        activity_freq = _compute_activity_freq(event_log.df)
        avg_time_from = _compute_avg_time_from_activity(dfg_map)
        pm = _build_process_model_from_petri(net, im, fm, noise_threshold, dfg_map=dfg_map, activity_freq=activity_freq, avg_time_from=avg_time_from, event_log_df=event_log.df)
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
        pm.conformance_metrics = _compute_conformance_metrics(event_log, net, im, fm)
        return pm

class DirectlyFollowsGraphEngine:
    def discover(self, event_log: EventLog, include_performance: bool = True) -> DirectlyFollowsGraph:
        """Discover a Directly-Follows Graph using pm4py's native DFG discovery.

        Args:
            event_log: EventLog containing a pandas DataFrame (`event_log.df`). Must include
                at least 'case:concept:name' and 'concept:name'. If 'time:timestamp' is present
                and include_performance is True, performance metrics (avg time between activities)
                will be included when available.
            include_performance: Whether to attempt performance variant discovery (requires timestamps).

        Returns:
            DirectlyFollowsGraph with `edges` populated as a list of dicts:
            [{ source, target, frequency, performance_seconds }]. performance_seconds defaults to 0.0 if unavailable.
            Start and end activities are attached as dynamic attributes (`start_activities`, `end_activities`).
        """
        pm = DirectlyFollowsGraph()
        edges, start_acts, end_acts = _compute_dfg(df=event_log.df, include_performance=include_performance)
        pm.edges = edges
        # attach start/end activities if discovered (not defined in dataclass but useful for callers)
        setattr(pm, 'start_activities', start_acts)
        setattr(pm, 'end_activities', end_acts)
        # build node list from unique activities (for cytoscape convenience)
        pm.nodes = _compute_dfg_nodes(event_log.df)
        # activities = set()
        # for e in edges:
        #     activities.add(e['source'])
        #     activities.add(e['target'])
        # pm.nodes = [{'data': {'id': a, 'label': a, 'type': 'activity'}} for a in sorted(activities)]
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
        else:
            data['frequency'] = 0
        if performance_seconds is not None:
            try:
                data['performance_seconds'] = float(performance_seconds)
            except Exception:
                data['performance_seconds'] = performance_seconds
        else:
            data['performance_seconds'] = 0.0

        # build label combining frequency and performance for Cytoscape edge display
        # ================= IN DEVELOPMENT =================
        # NOT DECIDED YET IMPLEMENTED OR NOT
        try:
            label_parts = []
            if 'frequency' in data and data['frequency'] is not None:
                try:
                    label_parts.append(str(int(data['frequency'])))
                except Exception:
                    label_parts.append(str(data['frequency']))
            if 'performance_seconds' in data and data['performance_seconds'] is not None:
                try:
                    label_parts.append(f"{float(data['performance_seconds']):.2f}s")
                except Exception:
                    label_parts.append(str(data['performance_seconds']))
            data['label'] = " / ".join(label_parts) if label_parts else ""
        except Exception:
            data['label'] = ""

        edges.append({
            'data': data
        })
        # ================= END IN DEVELOPMENT =================

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

    # Compute Start Time
    if event_log_df is not None and 'time:timestamp' in event_log_df.columns:
        try:
            el_df['time:timestamp'] = pd.to_datetime(el_df['time:timestamp'], errors='coerce')
            min_time = el_df['time:timestamp'].min()
            max_time = el_df['time:timestamp'].max()
            if not pd.isna(min_time):
                model_stats['start_time'] = pd.Timestamp(min_time).strftime("%Y-%m-%d %H:%M:%S.%f")
            else:
                model_stats['start_time'] = None
            if not pd.isna(max_time):
                model_stats['end_time'] = pd.Timestamp(max_time).strftime("%Y-%m-%d %H:%M:%S.%f")
            else:
                model_stats['end_time'] = None
        except Exception:
            model_stats['start_time'] = None
            model_stats['end_time'] = None
    if event_log_df is not None and 'concept:name' in event_log_df.columns:
        try:
            model_stats['number_of_events'] = int(len(event_log_df))
            case_durations = []
            grouped = el_df.groupby('case:concept:name')
            for case_id, group in grouped:
                times = group['time:timestamp'].dropna()
                if len(times) >= 2:
                    duration = (times.max() - times.min()).total_seconds()
                    case_durations.append(duration)
            if case_durations:
                model_stats['median_case_duration_seconds'] = int(pd.Series(case_durations).median())
                model_stats['mean_case_duration_seconds'] = int(pd.Series(case_durations).mean())
            else:
                model_stats['median_case_duration_seconds'] = None
                model_stats['mean_case_duration_seconds'] = None
        except Exception:
            model_stats['number_of_events'] = 0
    model_stats['number_of_cases'] = int(number_of_cases)
    model_stats['number_of_variants'] = int(number_of_variants)

    return ProcessModel(nodes=nodes, edges=edges, model_statistics=model_stats)

def _compute_dfg(df: pd.DataFrame, include_performance: bool = True) -> tuple[List[Dict[str, object]], List[str], List[str]]:
    """Compute directly-follows metrics using pm4py's DFG algorithm.

    Returns:
        (edges_list, start_activities, end_activities)
        edges_list: list of dicts { source, target, frequency, median, mean, median_string, mean_string, level }
    Notes:
        - If timestamps are missing or performance discovery fails, performance_seconds defaults to 0.0.
        - Robust against differing pm4py return structures for PERFORMANCE variant.
    """
    edges: List[Dict[str, object]] = []
    start_activities: List[str] = []
    end_activities: List[str] = []

    if df is None or 'case:concept:name' not in df.columns or 'concept:name' not in df.columns:
        return edges, start_activities, end_activities

    # Normalize timestamps if present
    has_time = 'time:timestamp' in df.columns
    if has_time:
        try:
            df = df.copy()
            df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], errors='coerce')
            if not df['time:timestamp'].notnull().any():
                has_time = False
        except Exception:
            has_time = False

    if include_performance and has_time:
        try:
            perf, start_activities, end_activities = pm4py.discover_performance_dfg(df)
        except Exception:
            perf = None

    try:
        freq_dfg, start_freq, end_freq = pm4py.discover_dfg(df)
    except Exception:
        freq_dfg = None

    # Build a set of all directly-follows pairs seen in either perf or freq
    pairs = set()
    if perf:
        try:
            pairs.update(perf.keys())
        except Exception:
            pass
    if freq_dfg:
        try:
            pairs.update(freq_dfg.keys())
        except Exception:
            pass

    # Merge performance and frequency information
    for (source, target) in pairs:
        perf_stats = perf.get((source, target)) if perf else None
        freq_cnt = None
        if freq_dfg:
            try:
                freq_cnt = int(freq_dfg.get((source, target), 0))
            except Exception:
                freq_cnt = freq_dfg.get((source, target), 0)

        # Performance metrics (default to zeros if missing)
        median = 0
        mean = 0.0
        if perf_stats:
            median = perf_stats.get('median', 0)
            mean = perf_stats.get('mean', 0.0)

        m1, s1 = divmod(int(median), 60)
        m2, s2 = divmod(int(mean), 60)
        median_string = f"{m1}M {s1}S"
        mean_string = f"{m2}M {s2}S"

        edges.append({
            'source': source,
            'target': target,
            'frequency': int(freq_cnt) if isinstance(freq_cnt, (int, float)) else (freq_cnt or 0),
            'median': int(median),
            'median_string': median_string,
            'mean': float(mean),
            'mean_string': mean_string,
        })

    if edges:
        freqs = [e['frequency'] for e in edges]
        medians = [e['median'] for e in edges]
        means = [e['mean'] for e in edges]
        max_freq = max(freqs) if freqs else 1
        max_median = max(medians) if medians else 1
        max_mean = max(means) if means else 1

        def classify(score: float) -> str:
            if score <= 0.33:
                return 'rendah'
            elif score <= 0.66:
                return 'sedang'
            return 'tinggi'

        for e in edges:
            nf = (e['frequency'] / max_freq) if max_freq > 0 else 0.0
            nm = (e['median'] / max_median) if max_median > 0 else 0.0
            nmean = (e['mean'] / max_mean) if max_mean > 0 else 0.0
            e['level_frequency'] = classify(nf)
            e['level_median'] = classify(nm)
            e['level_mean'] = classify(nmean)
        # (Opsional) jika ingin tetap mempertahankan field lama 'level', bisa dihapus baris berikut kalau tidak perlu.
        # e['level'] sekarang didefinisikan ulang sebagai gabungan sederhana (mayoritas dari tiga level)
        # Majority vote: jika dua atau lebih sama, pakai itu; kalau semua beda, pakai level_frequency.
        for e in edges:
            levels = [e['level_frequency'], e['level_median'], e['level_mean']]
            counts = {l: levels.count(l) for l in set(levels)}
            majority = max(counts.items(), key=lambda x: x[1])[0]
            e['level'] = majority

    return edges, start_activities, end_activities

def _compute_dfg_nodes(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Hitung frekuensi kemunculan setiap aktivitas dalam log.

    Mengimplementasikan algoritma manual yang menghitung langsung isi kolom aktivitas:

        nama_kolom_aktivitas = 'concept:name'
        frekuensi_aktivitas = {}
        for aktivitas in df[nama_kolom_aktivitas]:
            if aktivitas in frekuensi_aktivitas:
                frekuensi_aktivitas[aktivitas] += 1
            else:
                frekuensi_aktivitas[aktivitas] = 1

    Return:
        List[Dict] -> [{ 'activity': <nama>, 'count': <frekuensi> }, ...] diurutkan menurun berdasarkan count.

    Catatan:
        - Jika kolom 'concept:name' tidak ada, mengembalikan list kosong.
        - Nilai NaN akan di-skip.
    """
    if df is None or 'concept:name' not in df.columns:
        return []

    freq: Dict[str, int] = {}
    for aktivitas in df['concept:name']:
        if pd.isna(aktivitas):
            continue
        if aktivitas in freq:
            freq[aktivitas] += 1
        else:
            freq[aktivitas] = 1

    # susun list hasil, urutkan berdasarkan frekuensi menurun
    hasil = [
        {'activity': nama, 'count': int(jumlah)}
        for nama, jumlah in sorted(freq.items(), key=lambda x: x[1], reverse=True)
    ]
    return hasil

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