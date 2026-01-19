import csv
import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.log.exporter.xes import exporter as xes_exporter


class Converter:

    # @staticmethod
    # def csv_to_log(csv_path, output_path="event_log.xes.gz"):
    #     df = pd.read_csv(csv_path)
    #     print(df.head())
    #     timestamp_candidates = [
    #         'timestamp',
    #         'time:timestamp',
    #         'time',
    #         'event_time',
    #         'datetime'
    #     ]
    #     timestamp_col = next((col for col in timestamp_candidates if col in df.columns), None)
    #     if timestamp_col is None:
    #         raise ValueError("CSV must contain a timestamp column (timestamp, time:timestamp, time, event_time, datetime)")

    #     df['time:timestamp'] = pd.to_datetime(df[timestamp_col], errors='coerce')
    #     if df['time:timestamp'].isna().all():
    #         raise ValueError("Unable to parse timestamp column into valid datetime values")

    #     event_log = log_converter.apply(df)
    #     xes_exporter.apply(event_log, output_path)

    #     return event_log, output_path
    
    @staticmethod
    def xes_to_log(xes_path, output_path="event_log.xes.gz"):
        event_log = xes_importer.apply(xes_path)
        xes_exporter.apply(event_log, output_path)
        return event_log, output_path

    @staticmethod
    def csv_to_log(csv_path, output_path="event_log.xes.gz"):
        delimiter = ','
        try:
            with open(csv_path, 'r', encoding='utf-8') as csv_file:
                sample = csv_file.read(4096)
                csv_file.seek(0)
                delimiter = csv.Sniffer().sniff(sample, delimiters=',;\t|').delimiter
        except Exception:
            # fallback to semicolon if comma detection fails (common in EU datasets)
            delimiter = ';'

        df = pd.read_csv(csv_path, sep=delimiter)
        original_columns = df.columns.tolist()
        df.columns = [str(col).lower().strip() for col in df.columns]

        alias_map = {
            'case id': 'case:concept:name',
            'case_id': 'case:concept:name',
            'case': 'case:concept:name',
            'case concept name': 'case:concept:name',
            'activity': 'concept:name',
            'concept:name': 'concept:name',
            'event': 'concept:name',
            'resource': 'org:resource',
            'org:resource': 'org:resource',
            'role': 'org:resource',
            'cost': 'costs',
            'costs': 'costs',
        }
        timestamp_aliases = [
            'timestamp',
            'time:timestamp',
            'time',
            'event_time',
            'datetime',
            'complete timestamp',
            'complete_timestamp'
        ]

        normalized_cols = {col: col for col in df.columns}
        for idx, original in enumerate(original_columns):
            normalized_cols[original.lower().strip()] = df.columns[idx]

        for source_alias, target in alias_map.items():
            if source_alias in normalized_cols and target not in df.columns:
                df[target] = df[normalized_cols[source_alias]]

        timestamp_col = next(
            (
                normalized_cols[name]
                for name in timestamp_aliases
                if name in normalized_cols
            ),
            None
        )
        if timestamp_col is None:
            raise ValueError(
                "CSV must contain a recognizable timestamp column (timestamp, time:timestamp, time, event_time, datetime)"
            )

        df['time:timestamp'] = pd.to_datetime(df[timestamp_col], errors='coerce')
        if df['time:timestamp'].isna().all():
            raise ValueError("Unable to parse timestamp column into valid datetime values")

        required_columns = ['case:concept:name', 'concept:name']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"CSV must contain '{col}' column or an alias (case_id/activity)")

        event_log = log_converter.apply(df)
        xes_exporter.apply(event_log, output_path)

        return event_log, output_path

