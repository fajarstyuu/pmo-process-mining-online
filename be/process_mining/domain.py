from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd
import io
import gzip
import tempfile
import os
import pm4py


def _xes_with_rust(path: str, xes_bytes: bytes | None = None):
    """Try to read XES using rustxes (if installed) for speed, otherwise fall back to pm4py.read_xes.

    The function attempts several likely rustxes API names and falls back gracefully.
    """
    try:
        import rustxes as rx  # type: ignore
    except Exception:
        return pm4py.read_xes(path)

    # attempt a few possible rustxes entrypoints
    # prefer functions that accept bytes/string when xes_bytes is provided
    attempts = []
    if xes_bytes is not None:
        try:
            text = xes_bytes.decode('utf-8')
        except Exception:
            text = None
        if text is not None:
            attempts.extend([
                ('import_from_string', (text,)),
                ('loads', (text,)),
                ('parse', (text,)),
            ])
    # file-path based attempts
    attempts.extend([
        ('import_from_file', (path,)),
        ('import_log', (path,)),
        ('parse_file', (path,)),
        ('parse', (path,)),
    ])

    for name, args in attempts:
        func = getattr(rx, name, None)
        if callable(func):
            try:
                return func(*args)
            except Exception:
                # try next
                continue

    # last resort: let pm4py handle it
    return pm4py.read_xes(path)


@dataclass
class Event:
    case_id: str
    activity: str
    timestamp: Optional[pd.Timestamp] = None
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EventLog:
    # Keep underlying dataframe to leverage pm4py and pandas functions
    df: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)
    log: Any = None

    def number_of_cases(self) -> int:
        return int(self.df['case:concept:name'].nunique())

    def number_of_events(self) -> int:
        return int(len(self.df))

    def activities(self) -> List[str]:
        return self.df['concept:name'].unique().tolist()


@dataclass
class ProcessModel:
    # Simple representation for front-end consumption
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    model_statistics: Dict[str, Any] = field(default_factory=dict)
    # optional directly-follows graph metrics (separate from Petri net arcs)
    dfg: List[Dict[str, Any]] = field(default_factory=list)
    conformance_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_cytoscape(self) -> Dict[str, Any]:
        return {
            'nodes': self.nodes,
            'edges': self.edges,
            'model_statistics': self.model_statistics,
            'conformance_metrics': self.conformance_metrics,
        }

class DirectlyFollowsGraph:
    """Representation of a Directly-Follows Graph (DFG) with frequencies."""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]

    def __init__(self):
        self.edges = []
        self.nodes = []

    def add_edge(self, from_activity: str, to_activity: str, frequency: int = 1):
        key = (from_activity, to_activity)
        if key in self.edges:
            self.edges[key] += frequency
        else:
            self.edges[key] = frequency

    def to_list(self) -> List[Dict[str, Any]]:
        return [{'from': k[0], 'to': k[1], 'frequency': v} for k, v in self.edges.items()]
    
    def to_cytoscape(self) -> Dict[str, Any]:
        return {
            'nodes': self.nodes,
            'edges': self.edges
        }

class LogParser:
    """
    Parse uploaded files (CSV or XES) and produce EventLog instances.
    This isolates file parsing from discovery logic.
    """

    def parse(self, uploaded_file) -> EventLog:
        filename = getattr(uploaded_file, 'name', '')
        if not filename:
            raise ValueError('Uploaded file must have a filename')

        file_extension = filename.lower().split('.')[-1]
        content = uploaded_file.read()

        # support .csv, .xes and compressed .xes.gz
        if file_extension == 'csv':
            # decode and create DataFrame
            try:
                df = pd.read_csv(io.StringIO(content.decode('utf-8')))
            except Exception as e:
                raise ValueError(f'Unable to parse CSV: {str(e)}')

            # Try to normalize columns used by pm4py
            mapping = {}
            cols = list(df.columns)
            lower_map = {c.lower(): c for c in cols}

            # common alternatives
            alternatives = {
                'case:concept:name': ['case:concept:name', 'case_id', 'caseid', 'case', 'trace_id', 'case id'],
                'concept:name': ['concept:name', 'activity', 'event', 'activity_name', 'task'],
                'time:timestamp': ['time:timestamp', 'timestamp', 'time', 'datetime', 'date']
            }

            for target, alts in alternatives.items():
                found = None
                for alt in alts:
                    if alt in lower_map:
                        found = lower_map[alt]
                        break
                if found:
                    mapping[found] = target
                else:
                    # try fuzzy by contains
                    for low, orig in lower_map.items():
                        for alt in alts:
                            if alt.replace(':', '').replace('_', '') in low.replace('_', '').replace(':', ''):
                                mapping[orig] = target
                                found = orig
                                break
                        if found:
                            break

            if mapping:
                df = df.rename(columns=mapping)

            # Ensure required columns exist
            required = ['case:concept:name', 'concept:name']
            for r in required:
                if r not in df.columns:
                    raise ValueError(f"Required column '{r}' not found in CSV")

            # If timestamp exists, ensure datetime
            if 'time:timestamp' in df.columns:
                df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], errors='coerce')

            # Format for pm4py
            try:
                event_log = pm4py.format_dataframe(df, case_id='case:concept:name', activity_key='concept:name', timestamp_key='time:timestamp' if 'time:timestamp' in df.columns else None)
            except Exception:
                # Some versions of pm4py require timestamp key to exist; if not, pass without timestamp
                event_log = pm4py.format_dataframe(df, case_id='case:concept:name', activity_key='concept:name')

            metadata = {
                'number_of_cases': int(event_log['case:concept:name'].nunique()),
                'number_of_events': int(len(event_log))
            }

            return EventLog(df=event_log, metadata=metadata)

        elif filename.lower().endswith('.xes.gz') or file_extension == 'gz':
            # content is gzipped XES; decompress then feed to pm4py via a temp .xes file
            try:
                xes_bytes = gzip.decompress(content)
            except Exception as e:
                raise ValueError(f'Unable to decompress .xes.gz: {e}')

            with tempfile.NamedTemporaryFile(delete=False, suffix='.xes') as tmp:
                tmp.write(xes_bytes)
                tmp_path = tmp.name
            try:
                # try to use rustxes if available for faster import, fallback to pm4py
                event_log = _xes_with_rust(tmp_path, xes_bytes=xes_bytes)
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

            try:
                df = pm4py.convert_to_dataframe(event_log)
            except Exception:
                df = pd.DataFrame()

            metadata = {'number_of_cases': len(event_log)}
            return EventLog(df=df, metadata=metadata, log=event_log)

        elif file_extension == 'xes':
            # write to a temp file and use pm4py.read_xes
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xes') as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            try:
                event_log = _xes_with_rust(tmp_path, xes_bytes=content)
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

            # pm4py returns an Event Log object; transform to DataFrame for convenience
            try:
                df = pm4py.convert_to_dataframe(event_log)
            except Exception:
                # If conversion fails, keep a minimal DataFrame
                df = pd.DataFrame()

            metadata = {'number_of_cases': len(event_log)}
            return EventLog(df=df, metadata=metadata, log=event_log)

        else:
            raise ValueError('Unsupported file format. Use CSV or XES.')

class EventLogFilters:
    """Utility filters that operate on EventLog objects.

    Two filters implemented:
    - variants_coverage(event_log, coverage_target): keep traces (cases) whose
      variants (sequence of activities) cumulatively cover at least coverage_target of traces.
    - events_coverage(event_log, coverage_target): keep events whose activities (concept:name)
      cumulatively cover at least coverage_target of events.

    These implementations use pandas operations on EventLog.df and do not
    require pm4py filtering helpers (so they're robust even if pm4py filtering APIs differ).
    """

    @staticmethod
    def variants_coverage(event_log: EventLog, coverage_target: float = 1.0) -> EventLog:
        """Return an EventLog filtered to a minimal set of variants that cover
        at least `coverage_target` fraction of traces (cases).

        Args:
            event_log: EventLog to filter (expects 'case:concept:name' and 'concept:name').
            coverage_target: float in (0,1], fraction of total traces to cover.

        Returns:
            EventLog containing only traces whose variant is in the selected set.
        """
        if coverage_target <= 0:
            return EventLog(df=event_log.df.copy(), metadata=event_log.metadata.copy())

        df = event_log.df
        if df is None or 'case:concept:name' not in df.columns or 'concept:name' not in df.columns:
            return EventLog(df=pd.DataFrame(columns=df.columns) if df is not None else pd.DataFrame(), metadata={'number_of_cases': 0, 'number_of_events': 0})

        # prepare timestamps if available
        has_time = 'time:timestamp' in df.columns
        if has_time:
            try:
                df = df.copy()
                df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], errors='coerce')
                has_time = df['time:timestamp'].notnull().any()
            except Exception:
                has_time = False

        grouped = df.groupby('case:concept:name')

        # Build mapping case_id -> variant (tuple of activities), and variant counts
        case_variant = {}
        variant_counts: Dict[tuple, int] = {}
        for case_id, group in grouped:
            if has_time:
                try:
                    group = group.sort_values('time:timestamp')
                except Exception:
                    group = group.sort_index()
            else:
                group = group.sort_index()

            seq = tuple(group['concept:name'].tolist())
            case_variant[case_id] = seq
            variant_counts[seq] = variant_counts.get(seq, 0) + 1

        total_traces = sum(variant_counts.values())
        if total_traces == 0:
            return EventLog(df=pd.DataFrame(columns=df.columns), metadata={'number_of_cases': 0, 'number_of_events': 0})

        # sort variants by frequency desc
        variant_freq_list = sorted([(variant, cnt) for variant, cnt in variant_counts.items()], key=lambda x: x[1], reverse=True)

        selected_variants = []
        running = 0
        for variant, freq in variant_freq_list:
            selected_variants.append(variant)
            running += freq
            if running / total_traces >= coverage_target:
                break

        # select cases that have variant in selected_variants
        selected_cases = [case_id for case_id, var in case_variant.items() if var in selected_variants]

        filtered_df = df[df['case:concept:name'].isin(selected_cases)].copy()

        metadata = {
            'number_of_cases': int(filtered_df['case:concept:name'].nunique()) if 'case:concept:name' in filtered_df.columns else 0,
            'number_of_events': int(len(filtered_df))
        }
        return EventLog(df=filtered_df, metadata=metadata)

    @staticmethod
    def events_coverage(event_log: EventLog, coverage_target: float = 1.0) -> EventLog:
        """Return an EventLog filtered to activities that cumulatively cover at least
        `coverage_target` fraction of events.

        Args:
            event_log: EventLog (expects 'concept:name').
            coverage_target: float in (0,1], fraction of events to cover.

        Returns:
            EventLog with only events whose activity is in the selected set.
        """
        if coverage_target <= 0:
            return EventLog(df=event_log.df.copy(), metadata=event_log.metadata.copy())

        df = event_log.df
        if df is None or 'concept:name' not in df.columns:
            return EventLog(df=pd.DataFrame(columns=df.columns) if df is not None else pd.DataFrame(), metadata={'number_of_cases': 0, 'number_of_events': 0})

        activity_freq = df['concept:name'].value_counts().to_dict()
        total_events = sum(activity_freq.values())
        if total_events == 0:
            return EventLog(df=pd.DataFrame(columns=df.columns), metadata={'number_of_cases': 0, 'number_of_events': 0})

        sorted_activities = sorted(activity_freq.items(), key=lambda x: x[1], reverse=True)

        selected_activities = []
        running = 0
        for act, freq in sorted_activities:
            selected_activities.append(act)
            running += freq
            if running / total_events >= coverage_target:
                break

        filtered_df = df[df['concept:name'].isin(selected_activities)].copy()

        metadata = {
            'number_of_cases': int(filtered_df['case:concept:name'].nunique()) if 'case:concept:name' in filtered_df.columns else 0,
            'number_of_events': int(len(filtered_df))
        }

        return EventLog(df=filtered_df, metadata=metadata)

    @staticmethod
    def case_duration(event_log: EventLog, min_performance: float, max_performance: float) -> EventLog:
        """Add 'case:duration' attribute to each event in the EventLog DataFrame,
        representing the duration of the case (trace) in seconds.

        Args:
            event_log: EventLog with 'case:concept:name' and 'time:timestamp'.
        """

        df = event_log.df
        if df is None or 'case:concept:name' not in df.columns or 'time:timestamp' not in df.columns:
            return event_log
        
        try:
            log_filtered_duration = pm4py.filter_case_performance(df, min_performance=min_performance, max_performance=max_performance, timestamp_key='time:timestamp', case_id_key='case:concept:name')
            metadata = {
                'number_of_cases': int(log_filtered_duration['case:concept:name'].nunique()),
                'number_of_events': int(len(log_filtered_duration))
            }
            return EventLog(df=log_filtered_duration, metadata=metadata)
        except Exception:
            return event_log

    def case_size(event_log: EventLog, min_size: int, max_size: int) -> EventLog:
        """Filter EventLog to only include cases (traces) whose size (number of events)
        is within [min_size, max_size].

        Args:
            event_log: EventLog with 'case:concept:name'.
        """

        df = event_log.df
        if df is None or 'case:concept:name' not in df.columns:
            return event_log

        try:
            log_filtered_size = pm4py.filter_case_size(df, min_size=min_size, max_size=max_size, case_id_key='case:concept:name')
            metadata = {
                'number_of_cases': int(log_filtered_size['case:concept:name'].nunique()),
                'number_of_events': int(len(log_filtered_size))
            }
            return EventLog(df=log_filtered_size, metadata=metadata)
        except Exception:
            return event_log