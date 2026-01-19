from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class FilterConfiguration:
    event_log: Optional[Any] = None
    variant_coverage: float = 1.0
    event_coverage: float = 1.0
    start_time_performance: Optional[float] = None
    end_time_performance: Optional[float] = None
    max_size_performance: Optional[float] = None
    min_size_performance: Optional[float] = None

    def get_event_log(self):
        return self.event_log

    def get_variant_coverage(self):
        return self.variant_coverage

    def get_event_coverage(self):
        return self.event_coverage

    def get_start_time_performance(self):
        return self.start_time_performance

    def get_end_time_performance(self):
        return self.end_time_performance

    def get_max_size_performance(self):
        return self.max_size_performance

    def get_min_size_performance(self):
        return self.min_size_performance

    def set_event_log(self, event_log):
        self.event_log = event_log

    def set_variant_coverage(self, variant_coverage):
        self.variant_coverage = variant_coverage

    def set_event_coverage(self, event_coverage):
        self.event_coverage = event_coverage

    def set_start_time_performance(self, start_time_performance):
        self.start_time_performance = start_time_performance

    def set_end_time_performance(self, end_time_performance):
        self.end_time_performance = end_time_performance

    def set_max_size_performance(self, max_size_performance):
        self.max_size_performance = max_size_performance

    def set_min_size_performance(self, min_size_performance):
        self.min_size_performance = min_size_performance

    def set_all(self,
               event_log,
               variant_coverage,
               event_coverage,
               start_time_performance,
               end_time_performance,
               max_size_performance,
               min_size_performance):
        self.event_log = event_log
        self.variant_coverage = variant_coverage
        self.event_coverage = event_coverage
        self.start_time_performance = start_time_performance
        self.end_time_performance = end_time_performance
        self.max_size_performance = max_size_performance
        self.min_size_performance = min_size_performance

    def get_all(self) -> Dict[str, Any]:
        return {
            'event_log': self.event_log,
            'variant_coverage': self.variant_coverage,
            'event_coverage': self.event_coverage,
            'start_time_performance': self.start_time_performance,
            'end_time_performance': self.end_time_performance,
            'max_size_performance': self.max_size_performance,
            'min_size_performance': self.min_size_performance,
        }