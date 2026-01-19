from typing import Any, Dict, List, Optional

from process_mining.model.ProcessModel import ProcessModel


class ProcessModelNotLoadedError(RuntimeError):
    """Raised when no ProcessModel has been set."""


class ProcessModelService:
    """Lightweight setter/getter wrapper for ProcessModel objects."""

    def __init__(self, process_model: Optional[ProcessModel] = None) -> None:
        self._process_model = process_model

    @property
    def is_loaded(self) -> bool:
        return self._process_model is not None

    def set_process_model(self, process_model: ProcessModel) -> None:
        if process_model is None:
            raise ValueError("process_model cannot be None")
        self._process_model = process_model

    def set_nodes(self, nodes: List[Dict[str, Any]]) -> None:
        self._process_model.set_nodes(nodes)

    def set_edges(self, edges: List[Dict[str, Any]]) -> None:
        self._process_model.set_edges(edges)

    def set_model_statistics(self, model_statistics: Dict[str, Any]) -> None:
        self._process_model.set_model_statistics(model_statistics)

    def set_evaluation_metrics(self, conformance_metrics: Dict[str, Any]) -> None:
        self._process_model.set_evaluation_metrics(conformance_metrics)

    def get_nodes(self) -> List[Dict[str, Any]]:
        return self._process_model.get_nodes()

    def get_edges(self) -> List[Dict[str, Any]]:
        return self._process_model.get_edges()

    def get_model_statistics(self) -> Dict[str, Any]:
        return self._process_model.get_model_statistics()

    def get_evaluation_metrics(self) -> Dict[str, Any]:
        return self._process_model.get_evaluation_metrics()

    def to_cytoscape(self) -> Dict[str, Any]:
        """Return a cytoscape-serializable dict for the current process model.

        Ensures a ProcessModel exists, then serializes its nodes/edges/statistics.
        """
        return {
            'nodes': self._process_model.get_nodes(),
            'edges': self._process_model.get_edges(),
            'model_statistics': self._process_model.get_model_statistics(),
            'conformance_metrics': self._process_model.get_evaluation_metrics(),
        }

    def build_process_model_from_petri(self, petri_net, initial_marking, final_marking, noise_threshold: float = 0.0, dfg_map: Dict[tuple, dict] = None, activity_freq: Dict[str, int] = None, avg_time_from: Dict[str, float] = None, event_log_df: Optional[Any] = None) -> ProcessModel:
        """Build nodes/edges/statistics from a Petri net and attach to ProcessModel.

        This method mirrors the previous implementation that lived on the ProcessModel
        dataclass, but keeps the mutation logic inside the service wrapper.
        Returns the underlying ProcessModel instance.
        """
        # import pandas lazily to avoid circular import at module import time
        import pandas as _pd

        # ensure we have a model to attach to
        if not self.is_loaded:
            self.create()

        # replicate original build logic (compute nodes, edges, model_stats)
        nodes = []
        edges = []

        place_id_map = {}
        trans_id_map = {}

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

        for idx, arc in enumerate(petri_net.arcs):
            src_name = getattr(arc.source, 'name', None) or str(id(arc.source))
            tgt_name = getattr(arc.target, 'name', None) or str(id(arc.target))

            src_id = place_id_map.get(src_name) or trans_id_map.get(src_name) or f"unknown_{src_name}"
            tgt_id = place_id_map.get(tgt_name) or trans_id_map.get(tgt_name) or f"unknown_{tgt_name}"

            src_label = getattr(arc.source, 'label', None) if hasattr(arc.source, 'label') or hasattr(arc.source, 'name') else None
            if src_label is None and hasattr(arc.source, 'label'):
                src_label = getattr(arc.source, 'label', None)
            tgt_label = getattr(arc.target, 'label', None) if hasattr(arc.target, 'label') or hasattr(arc.target, 'name') else None

            frequency = None
            performance_seconds = None

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

            edges.append({'data': data})

        model_stats = {
            'number_of_places': len(petri_net.places),
            'number_of_transitions': len(petri_net.transitions),
            'number_of_arcs': len(petri_net.arcs),
            'noise_threshold_used': noise_threshold
        }

        # compute cases/variants if event log provided
        number_of_cases = 0
        number_of_variants = 0
        if event_log_df is not None and 'case:concept:name' in event_log_df.columns and 'concept:name' in event_log_df.columns:
            try:
                el_df = event_log_df.copy()
                has_time = 'time:timestamp' in el_df.columns
                if has_time:
                    try:
                        el_df['time:timestamp'] = _pd.to_datetime(el_df['time:timestamp'], errors='coerce')
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
                number_of_cases = 0
                number_of_variants = 0

        if event_log_df is not None and 'time:timestamp' in event_log_df.columns:
            try:
                el_df['time:timestamp'] = _pd.to_datetime(el_df['time:timestamp'], errors='coerce')
                min_time = el_df['time:timestamp'].min()
                max_time = el_df['time:timestamp'].max()
                if not _pd.isna(min_time):
                    model_stats['start_time'] = _pd.Timestamp(min_time).strftime("%Y-%m-%d %H:%M:%S.%f")
                else:
                    model_stats['start_time'] = None
                if not _pd.isna(max_time):
                    model_stats['end_time'] = _pd.Timestamp(max_time).strftime("%Y-%m-%d %H:%M:%S.%f")
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
                    model_stats['median_case_duration_seconds'] = int(_pd.Series(case_durations).median())
                    model_stats['mean_case_duration_seconds'] = int(_pd.Series(case_durations).mean())
                else:
                    model_stats['median_case_duration_seconds'] = None
                    model_stats['mean_case_duration_seconds'] = None
            except Exception:
                model_stats['number_of_events'] = 0

        model_stats['number_of_cases'] = int(number_of_cases)
        model_stats['number_of_variants'] = int(number_of_variants)

        # attach to model
        self.set_nodes(nodes)
        self.set_edges(edges)
        self.set_model_statistics(model_stats)

