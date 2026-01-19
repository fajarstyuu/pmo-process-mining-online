from typing import Dict
import pandas as pd
from pm4py.algo.evaluation.replay_fitness.variants import token_replay as fitness_evaluator
from pm4py.algo.evaluation.precision.variants import etconformance_token as precision_evaluator
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
from process_mining.model.EventLog import EventLog

class ConformanceService:
    def apply(event_log: EventLog, petri_net, initial_marking, final_marking) -> Dict[str, float]:
        """
        Compute conformance metrics: fitness, precision, generalization, simplicity
        """
        metrics = {}

        # Fitness
        fitness = fitness_evaluator.apply(event_log.df, petri_net, initial_marking, final_marking)
        metrics['fitness'] = fitness

        # Precision
        precision = precision_evaluator.apply(event_log.df, petri_net, initial_marking, final_marking)
        metrics['precision'] = precision

        # Generalization
        generalization = generalization_evaluator.apply(event_log.df, petri_net, initial_marking, final_marking)
        metrics['generalization'] = generalization

        # Simplicity
        simplicity = simplicity_evaluator.apply(petri_net)
        metrics['simplicity'] = simplicity

        return metrics