from abc import ABC, abstractmethod
from dataclasses import dataclass
from process_mining.model.EventLog import EventLog
import pandas as pd
import pm4py
from typing import Dict

@dataclass
class EventLogFilter(ABC):
    variant_coverage: float
    event_coverage: float
    start_time_performance: float
    end_time_performance: float
    max_size_performance: float
    min_size_performance: float
    filter_type: str
    @abstractmethod
    def apply(self, event_log, config):
        pass

@dataclass
class VariantCoverageFilter(EventLogFilter):
    event_log: EventLog
    def apply(self):
        if self.variant_coverage <= 0:
            return EventLog(df=self.event_log.df.copy(), metadata=self.event_log.metadata.copy())

        df = self.event_log.df
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
            if running / total_traces >= self.variant_coverage:
                break

        # select cases that have variant in selected_variants
        selected_cases = [case_id for case_id, var in case_variant.items() if var in selected_variants]

        filtered_df = df[df['case:concept:name'].isin(selected_cases)].copy()

        metadata = {
            'number_of_cases': int(filtered_df['case:concept:name'].nunique()) if 'case:concept:name' in filtered_df.columns else 0,
            'number_of_events': int(len(filtered_df))
        }
        return EventLog(df=filtered_df, metadata=metadata)

@dataclass
class EventCoverageFilter(EventLogFilter):
    event_log: EventLog
    def apply(self):
        if self.event_coverage <= 0:
            return EventLog(df=self.event_log.df.copy(), metadata=self.event_log.metadata.copy())

        df = self.event_log.df
        if df is None or 'concept:name' not in df.columns:
            return EventLog(df=pd.DataFrame(columns=df.columns) if df is not None else pd.DataFrame(), metadata={'number_of_cases': 0, 'number_of_events': 0})

        activity_freq = df['concept:name'].value_counts().to_dict()
        total_events = sum(activity_freq.values())
        if total_events == 0:
            return EventLog(df=pd.DataFrame(columns=df.columns), metadata={'number_of_cases': 0, 'number_of_events': 0})

        sorted_activities = sorted(activity_freq.items(), key=lambda x: x[1], reverse=True)

        selected_activities = []
        running = 0
        for activity, freq in sorted_activities:
            selected_activities.append(activity)
            running += freq
            if running / total_events >= self.event_coverage:
                break

        filtered_df = df[df['concept:name'].isin(selected_activities)].copy()

        metadata = {
            'number_of_cases': int(filtered_df['case:concept:name'].nunique()) if 'case:concept:name' in filtered_df.columns else 0,
            'number_of_events': int(len(filtered_df))
        }
        return EventLog(df=filtered_df, metadata=metadata)

@dataclass
class CaseDurationFilter(EventLogFilter):
    event_log: EventLog
    def apply(self):
        df = self.event_log.df
        if df is None or 'case:concept:name' not in df.columns or 'time:timestamp' not in df.columns:
            return self.event_log

        try:
            log_filtered_duration = pm4py.filter_case_performance(
                df,
                min_performance=self.start_time_performance,
                max_performance=self.end_time_performance,
                timestamp_key='time:timestamp',
                case_id_key='case:concept:name'
            )
            metadata = {
                'number_of_cases': int(log_filtered_duration['case:concept:name'].nunique()),
                'number_of_events': int(len(log_filtered_duration))
            }
            return EventLog(df=log_filtered_duration, metadata=metadata)
        except Exception:
            return self.event_log

@dataclass
class CaseSizeFilter(EventLogFilter):
    event_log: EventLog
    def apply(self):
        df = self.event_log.df
        if df is None or 'case:concept:name' not in df.columns:
            return self.event_log

        try:
            log_filtered_size = pm4py.filter_case_size(
                df,
                min_size=self.min_size,
                max_size=self.max_size,
                case_id_key='case:concept:name'
            )
            metadata = {
                'number_of_cases': int(log_filtered_size['case:concept:name'].nunique()),
                'number_of_events': int(len(log_filtered_size))
            }
            return EventLog(df=log_filtered_size, metadata=metadata)
        except Exception:
            return self.event_log

