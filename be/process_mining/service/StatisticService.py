import pandas as pd
from typing import List, Dict, Any, Optional
from pm4py.statistics.variants.log import get as variants_module
from pm4py import get_event_attribute_values
import pm4py

class StatisticService:

    def compute_case_statistics(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
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

    def compute_events_statistics(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
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
            raise

    def compute_variant_statistics(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Compute variant statistics: frequency of each variant
        """
        """
        Return a list of variant statistics. Each item contains:
        {
            "variant": [<activity1>, <activity2>, ...],
            "count": int,
            "percent": float (rounded to 2 decimals)
        }

        Notes:
        - This function does NOT mutate the input DataFrame.
        - If case ids or activity names are missing, an empty list is returned.
        - Exceptions are not silently swallowed; errors will propagate to the caller.
        """
        if df is None or 'case:concept:name' not in df.columns or 'concept:name' not in df.columns:
            return []

        variants: List[Dict[Any, Any]] = []
        log_variant = variants_module.get_variants(df)
        for variant, count in log_variant.items():
            variant_string = " → ".join(variant)
            variants.append({
                "variant": variant_string,
                "count": len(count),
                "percent": round((len(count) / len(df['case:concept:name'].unique()) * 100), 2) if len(df['case:concept:name'].unique()) > 0 else 0.0
            })

        variants.sort(key=lambda x: x['count'], reverse=True)

        return variants

    def compute_resource_statistics(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Compute resource statistics: frequency of each resource
        """
        """
        Return a list of resource statistics. Each item contains:
        {
            "resource": <resource name>,
            "count": int,
            "percent": float (rounded to 2 decimals)
        }

        Notes:
        - This function does NOT mutate the input DataFrame.
        - If resource column is missing, an empty list is returned.
        - Exceptions are not silently swallowed; errors will propagate to the caller.
        """
        if df is None or 'org:resource' not in df.columns:
            return []

        try:
            resources: List[Dict[str, Any]] = []
            resource = get_event_attribute_values(df, "org:resource")
            event_percentage = (df['org:resource'].value_counts() / len(df) * 100).round(2) if len(df) > 0 else df['org:resource'].value_counts() * 0.0

            for res, count in resource.items():
                percent = float(event_percentage[res]) if res in event_percentage.index else 0.0
                resources.append({
                    "resource": res,
                    "count": count,
                    "percent": percent
                })

            return resources

        except Exception:
            raise

    def compute_model_statistics(self, petri_net, noise_threshold: float = 0.0, event_log_df: Optional[pd.DataFrame] = None, discover_when_missing: bool = True) -> Dict[str, Any]:
        """Aggregate structural and log-based metrics for a process model."""

        stats: Dict[str, Any] = {
            'number_of_places': len(getattr(petri_net, 'places', [])) if petri_net is not None else 0,
            'number_of_transitions': len(getattr(petri_net, 'transitions', [])) if petri_net is not None else 0,
            'number_of_arcs': len(getattr(petri_net, 'arcs', [])) if petri_net is not None else 0,
            'noise_threshold_used': noise_threshold,
            'number_of_cases': 0,
            'number_of_variants': 0,
            'number_of_events': 0,
            'start_time': None,
            'end_time': None,
            'median_case_duration_seconds': None,
            'mean_case_duration_seconds': None,
        }

        el_df: Optional[pd.DataFrame] = None
        if event_log_df is not None:
            try:
                el_df = event_log_df.copy()
            except Exception:
                try:
                    el_df = pd.DataFrame(event_log_df)
                except Exception:
                    el_df = None

        if petri_net is None and discover_when_missing and el_df is not None:
            try:
                discovered_net, _, _ = pm4py.discover_petri_net_inductive(el_df, noise_threshold=noise_threshold)
                petri_net = discovered_net
            except Exception:
                petri_net = None

            if petri_net is not None:
                stats['number_of_places'] = len(getattr(petri_net, 'places', []))
                stats['number_of_transitions'] = len(getattr(petri_net, 'transitions', []))
                stats['number_of_arcs'] = len(getattr(petri_net, 'arcs', []))

        if el_df is None:
            return stats

        stats['number_of_events'] = int(len(el_df))

        has_case_and_activity = {'case:concept:name', 'concept:name'}.issubset(el_df.columns)
        has_time = 'time:timestamp' in el_df.columns

        if has_time:
            el_df['time:timestamp'] = pd.to_datetime(el_df['time:timestamp'], errors='coerce')

        grouped_cases = None
        if has_case_and_activity:
            grouped_cases = el_df.groupby('case:concept:name')
            stats['number_of_cases'] = int(len(grouped_cases))
            variants = set()
            for _, group in grouped_cases:
                working_group = group
                if has_time and group['time:timestamp'].notnull().any():
                    working_group = group.sort_values('time:timestamp')
                else:
                    working_group = group.sort_index()
                variants.add(tuple(working_group['concept:name'].tolist()))
            stats['number_of_variants'] = int(len(variants))

        if has_time:
            min_time = el_df['time:timestamp'].min()
            max_time = el_df['time:timestamp'].max()
            stats['start_time'] = None if pd.isna(min_time) else pd.Timestamp(min_time).strftime("%Y-%m-%d %H:%M:%S.%f")
            stats['end_time'] = None if pd.isna(max_time) else pd.Timestamp(max_time).strftime("%Y-%m-%d %H:%M:%S.%f")

        if has_time and grouped_cases is not None:
            case_durations = []
            for _, group in grouped_cases:
                times = group['time:timestamp'].dropna()
                if len(times) >= 2:
                    case_durations.append((times.max() - times.min()).total_seconds())
            if case_durations:
                duration_series = pd.Series(case_durations)
                stats['median_case_duration_seconds'] = int(duration_series.median())
                stats['mean_case_duration_seconds'] = int(duration_series.mean())

        return stats
