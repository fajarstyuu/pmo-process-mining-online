import pandas as pd
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.objects.conversion.log import converter as log_converter

def compute_arc_performance_via_petri(event_log, net, im, fm,
case_id_col="case:concept:name",
activity_col="concept:name",
timestamp_col="time:timestamp"):
"""
Compute performance per Petri Net arc (transition → transition),
matching pm4py PERFORMANCE variant.
Accepts either EventLog or pandas DataFrame.
"""

    # --- 1. Convert EventLog to DataFrame if needed ---
    if not isinstance(event_log, pd.DataFrame):
        event_log = log_converter.apply(event_log, variant=log_converter.Variants.TO_DATA_FRAME)

    # pastikan sorting
    event_log = event_log.sort_values(by=[case_id_col, timestamp_col])


    # --- 2. Extract transitions with labels ---
    transition_arcs = []
    for t_src in net.transitions:
        if not t_src.label:
            continue
        # t_src → place → t_tgt
        for arc_out in t_src.out_arcs:
            p = arc_out.target
            for arc_in in p.out_arcs:
                t_tgt = arc_in.target
                if t_tgt.label:
                    transition_arcs.append((t_src.label, t_tgt.label))

    # --- 3. Prepare accumulator ---
    perf_store = {arc: {"s": 0.0, "c": 0} for arc in transition_arcs}

    # --- 4. Process per-case ordered events ---
    grouped = event_log.groupby(case_id_col)

    for _, case_df in grouped:
        events = case_df[[activity_col, timestamp_col]].values

        for i in range(len(events) - 1):
            a, t_a = events[i]
            b, t_b = events[i + 1]

            if (a, b) in perf_store:
                delta = (pd.to_datetime(t_b) - pd.to_datetime(t_a)).total_seconds()
                perf_store[(a, b)]["s"] += delta
                perf_store[(a, b)]["c"] += 1

    # --- 5. Compute averages ---
    final_perf = {}
    for key, v in perf_store.items():
        final_perf[key] = (v["s"] / v["c"]) if v["c"] > 0 else 0.0

    return final_perf

---

Update 2025-12-05:

- Fixed crash in `DiscoverModelAPIView` when optional numeric fields are empty strings.
  - Added safe parsers `_to_optional_float` and `_to_optional_int` that treat "" as None and handle bad inputs.
  - Applied guards so `case_size` and `case_duration` filters only run when both bounds are provided.
  - Ensured coverage filters use sane defaults when missing.
- Consider mirroring the same input handling in `ConformanceCheckAPIView` if users may submit empty fields.
