from process_mining.model.EventLog import EventLog
from process_mining.model.DirectlyFollowsGraph import DirectlyFollowsGraph
from typing import Any, List, Dict, Optional
import pm4py
import pandas as pd

class DirectlyFollowsGraphService:
    def discover(self, event_log: EventLog, include_performance: bool = True):
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
        edges, start_acts, end_acts = self.compute_dfg(df=event_log.df, include_performance=include_performance)
        nodes = self.compute_dfg_nodes(event_log.df)
        return edges, nodes
    
    def discover2(self, event_log: EventLog, include_performance: bool = True) -> DirectlyFollowsGraph:
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
        edges, start_acts, end_acts = self.compute_dfg(df=event_log.df, include_performance=include_performance)
        pm.edges = edges
        # attach start/end activities if discovered (not defined in dataclass but useful for callers)
        setattr(pm, 'start_activities', start_acts)
        setattr(pm, 'end_activities', end_acts)
        # build node list from unique activities (for cytoscape convenience)
        pm.nodes = self.compute_dfg_nodes(event_log.df)
        # activities = set()
        # for e in edges:
        #     activities.add(e['source'])
        #     activities.add(e['target'])
        # pm.nodes = [{'data': {'id': a, 'label': a, 'type': 'activity'}} for a in sorted(activities)]
        return pm
    
    def compute_dfg(self, df: pd.DataFrame, include_performance: bool = True) -> tuple[List[Dict[str, object]], List[str], List[str]]:
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
        perf = None

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

            def _format_days(seconds_value: float) -> str:
                days = (seconds_value or 0) / 86400.0
                formatted = f"{days:.1f}".replace('.', ',')
                return f"{formatted} hari"

            median_string = _format_days(float(median))
            mean_string = _format_days(float(mean))

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
    
    def compute_dfg_nodes(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
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
    
        hasil = [
            {'activity': nama, 'count': int(jumlah)}
            for nama, jumlah in sorted(freq.items(), key=lambda x: x[1], reverse=True)
        ]
        return hasil
