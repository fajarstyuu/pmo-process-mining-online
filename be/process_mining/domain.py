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
            'dfg': self.dfg,
            'conformance_metrics': self.conformance_metrics,
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
            return EventLog(df=df, metadata=metadata)

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
            return EventLog(df=df, metadata=metadata)

        else:
            raise ValueError('Unsupported file format. Use CSV or XES.')
