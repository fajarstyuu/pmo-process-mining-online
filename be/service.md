# Service Layer Reference

Dokumen ini merangkum seluruh class yang berada di folder `process_mining/service`. Setiap class dicantumkan bersama atribut utama (nama + tipe data + visibilitas) dan method-nya (signature + tipe data return + visibilitas). Penulisan "Private" merujuk pada nama yang diawali dengan `_`, sedangkan sisanya dianggap "Public".

---

## ConformanceService.py

### ConformanceService

**Attributes**

- _(none)_

**Methods**

- **[Public]** `apply(event_log: EventLog, petri_net, initial_marking, final_marking) -> Dict[str, float]`
  - Menghitung metrik conformance (fitness, precision, generalization, simplicity) menggunakan pm4py.

---

## DirectlyFollowsGraphService.py

### DirectlyFollowsGraphEngine (dataclass)

**Attributes**

- **[Public]** `event_log: EventLog` – log sumber yang akan dianalisis.
- **[Public]** `include_performance: bool = True` – flag untuk menyertakan metrik kinerja.

**Methods**

- **[Public]** `discover() -> DirectlyFollowsGraph` – menjalankan discovery DFG dan mengisi node/edge.
- **[Private]** `_compute_dfg(df: pd.DataFrame, include_performance: bool = True) -> tuple[List[Dict[str, object]], List[str], List[str]]` – menghitung edge + start/end activities.
- **[Private]** `_compute_dfg_nodes(df: pd.DataFrame) -> List[Dict[str, Any]]` – menghitung frekuensi aktivitas (node list).

### DirectlyFollowsGraphNotLoadedError

**Attributes / Methods**

- Turunan `RuntimeError` tanpa atribut tambahan; dipakai sebagai pengecualian ketika graph belum di-set.

### DirectlyFollowsGraphModelService

**Attributes**

- **[Private]** `_graph: Optional[DirectlyFollowsGraph]` – menyimpan graph yang sedang aktif.

**Methods**

- **[Public]** `__init__(graph: Optional[DirectlyFollowsGraph] = None) -> None`
- **[Public]** `is_loaded -> bool` (property) – status apakah `_graph` sudah terisi.
- **[Public]** `set_graph(graph: DirectlyFollowsGraph) -> None`
- **[Public]** `create(nodes: Optional[List[Dict[str, Any]]] = None, edges: Optional[List[Dict[str, Any]]] = None) -> DirectlyFollowsGraph`
- **[Public]** `update(**kwargs) -> DirectlyFollowsGraph` – menimpa atribut yang tersedia pada graph.
- **[Public]** `get_graph() -> DirectlyFollowsGraph`
- **[Public]** `clear() -> None`
- **[Private]** `_ensure_loaded() -> DirectlyFollowsGraph` – memastikan `_graph` tidak `None`.

---

## DiscoveryService.py

### DiscoveryService (ABC)

**Attributes**

- _(none)_

**Methods**

- **[Public | Abstract]** `discover(event_log: EventLog, noise_threshold: float = 0.0) -> ProcessModel`

### InductiveMiner / AlphaMiner / HeuristicMiner

**Attributes**

- _(none)_

**Methods**

- **[Public]** `discover(event_log: EventLog, noise_threshold: float = 0.0) -> tuple`
  - Masing-masing mengembalikan `net, im, fm` dari miner pm4py terkait.

---

## EventLogService.py

### EventLogNotLoadedError

- Turunan `RuntimeError` untuk penjaminan state EventLog.

### EventLogService

**Attributes**

- **[Private]** `_event_log: Optional[EventLog]`

**Methods**

- **[Public]** `__init__(event_log: Optional[EventLog] = None) -> None`
- **[Public]** `is_loaded -> bool` (property)
- **[Public]** `set_event_log(event_log: EventLog) -> None`
- **[Public]** `set_from_dataframe(df: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None, log: Any = None) -> None`
- **[Public]** `update_metadata(metadata: Dict[str, Any]) -> None`
- **[Public]** `clear() -> None`
- **[Public]** `get_event_log() -> EventLog`
- **[Public]** `get_dataframe() -> pd.DataFrame`
- **[Public]** `get_metadata() -> Dict[str, Any>`
- **[Public]** `get_underlying_log() -> Any`
- **[Private]** `_ensure_loaded() -> EventLog`

---

## FilterConfigurationService.py

### FilterConfigurationNotLoadedError

- Turunan `RuntimeError` tanpa atribut khusus.

### FilterConfigurationService

**Attributes**

- **[Private]** `_config: Optional[FilterConfiguration]`

**Methods**

- **[Public]** `__init__(config: Optional[FilterConfiguration] = None) -> None`
- **[Public]** `is_loaded -> bool` (property)
- **[Public]** `set_config(config: FilterConfiguration) -> None`
- **[Public]** `create(**kwargs) -> FilterConfiguration`
- **[Public]** `update(**kwargs) -> FilterConfiguration`
- **[Public]** `get_config() -> FilterConfiguration`
- **[Public]** `clear() -> None`
- **[Private]** `_ensure_loaded() -> FilterConfiguration`

---

## FilterService.py

### FilterService (ABC)

**Attributes**

- _(none)_

**Methods**

- **[Public | Abstract]** `apply(event_log: EventLog) -> EventLog`

### VariantCoverageFilter (dataclass)

**Attributes**

- **[Public]** `variant_coverage: float`

**Methods**

- **[Public]** `apply(event_log: EventLog) -> EventLog`

### EventCoverageFilter (dataclass)

**Attributes**

- **[Public]** `event_coverage: float`

**Methods**

- **[Public]** `apply(event_log: EventLog) -> EventLog`

### CaseDurationFilter (dataclass)

**Attributes**

- **[Public]** `start_time_performance: float`
- **[Public]** `end_time_performance: float`

**Methods**

- **[Public]** `apply(event_log: EventLog) -> EventLog`

### CaseSizeFilter (dataclass)

**Attributes**

- **[Public]** `min_size: int`
- **[Public]** `max_size: int`

**Methods**

- **[Public]** `apply(event_log: EventLog) -> EventLog`

---

## ProcessMiningService.py

### ProcessMiningNotLoadedError

- Turunan `RuntimeError`.

### ProcessMiningService

**Attributes**

- **[Private]** `_process_mining: Optional[ProcessMining]`

**Methods**

- **[Public]** `__init__(process_mining: Optional[ProcessMining] = None) -> None`
- **[Public]** `is_loaded -> bool` (property)
- **[Public]** `set_process_mining(process_mining: ProcessMining) -> None`
- **[Public]** `create(algorithm: str = "", parameters: Optional[Dict[str, Any]] = None, noise_threshold: float = 0.0, df: Optional[pd.DataFrame] = None, event_log: Any = None) -> ProcessMining`
- **[Public]** `update(**kwargs: Any) -> ProcessMining`
- **[Public]** `get_process_mining() -> ProcessMining`
- **[Public]** `clear() -> None`
- **[Private]** `_ensure_loaded() -> ProcessMining`

---

## ProcessModelService.py

### ProcessModelNotLoadedError

- Turunan `RuntimeError`.

### ProcessModelService

**Attributes**

- **[Private]** `_process_model: Optional[ProcessModel]`

**Methods**

- **[Public]** `__init__(process_model: Optional[ProcessModel] = None) -> None`
- **[Public]** `is_loaded -> bool` (property)
- **[Public]** `set_process_model(process_model: ProcessModel) -> None`
- **[Public]** `create(nodes: Optional[List[Dict[str, Any]]] = None, edges: Optional[List[Dict[str, Any]]] = None, model_statistics: Optional[Dict[str, Any]] = None, conformance_metrics: Optional[Dict[str, Any]] = None) -> ProcessModel`
- **[Public]** `update(**kwargs: Any) -> ProcessModel`
- **[Public]** `get_process_model() -> ProcessModel`
- **[Public]** `clear() -> None`
- **[Private]** `_ensure_loaded() -> ProcessModel`

---

## StatisticModelService.py

### StatisticNotLoadedError

- Turunan `RuntimeError`.

### StatisticModelService

**Attributes**

- **[Private]** `_statistic: Optional[Statistic]`

**Methods**

- **[Public]** `__init__(statistic: Optional[Statistic] = None) -> None`
- **[Public]** `is_loaded -> bool` (property)
- **[Public]** `set_statistic(statistic: Statistic) -> None`
- **[Public]** `create(events: Optional[List[Dict[str, Any]]] = None, case: Optional[List[Dict[str, Any]]] = None, variants: Optional[List[Dict[str, Any]]] = None, resources: Optional[List[Dict[str, Any]]] = None) -> Statistic`
- **[Public]** `update(**kwargs: Any) -> Statistic`
- **[Public]** `get_statistic() -> Statistic`
- **[Public]** `clear() -> None`
- **[Private]** `_ensure_loaded() -> Statistic`

---

## StatisticService.py

### StatisticService (dataclass)

**Attributes**

- **[Public]** `df: pd.DataFrame`

**Methods**

- **[Private]** `_compute_case_statistics() -> List[Dict[str, Any]]`
- **[Private]** `_compute_events_statistics() -> List[Dict[str, Any]]`
- **[Private]** `_compute_variant_statistics() -> List[Dict[str, Any]]`
- **[Private]** `_compute_resource_statistics() -> List[Dict[str, Any]]`

---

## Conformance-related / Filter utilities summary

- Semua class di atas berada di dalam folder `process_mining/service`. Gunakan tabel ini sebagai referensi cepat ketika membuat class diagram atau dokumentasi teknis lainnya.
