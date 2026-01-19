from django.test import TestCase
import pandas as pd

from process_mining.model.EventLog import EventLog
from process_mining.service.EventLogService import EventLogService, EventLogNotLoadedError
from process_mining.service.ProcessModelService import ProcessModelService, ProcessModelNotLoadedError
from process_mining.service.ProcessMiningService import ProcessMiningService, ProcessMiningNotLoadedError
from process_mining.service.FilterConfigurationService import FilterConfigurationService, FilterConfigurationNotLoadedError
from process_mining.service.StatisticModelService import StatisticModelService, StatisticNotLoadedError
from process_mining.service.DirectlyFollowsGraphService import (
	DirectlyFollowsGraphModelService,
	DirectlyFollowsGraphNotLoadedError,
)
from process_mining.model.Statistic import Statistic


class EventLogServiceTests(TestCase):
	def setUp(self) -> None:
		self.df = pd.DataFrame(
			[
				{"case:concept:name": "C1", "concept:name": "A"},
				{"case:concept:name": "C1", "concept:name": "B"},
				{"case:concept:name": "C2", "concept:name": "A"},
			]
		)

	def test_set_and_get_event_log(self):
		log = EventLog(df=self.df.copy(), metadata={'number_of_events': 3})
		service = EventLogService(event_log=log)
		self.assertEqual(service.get_event_log(), log)

	def test_set_from_dataframe(self):
		service = EventLogService()
		service.set_from_dataframe(self.df, metadata={'number_of_cases': 2})
		self.assertTrue(service.is_loaded)
		self.assertEqual(service.get_dataframe().shape[0], 3)
		self.assertEqual(service.get_metadata()['number_of_cases'], 2)

	def test_update_metadata(self):
		service = EventLogService()
		service.set_from_dataframe(self.df, metadata={'number_of_events': 3})
		service.update_metadata({'extra': 'value'})
		self.assertEqual(service.get_metadata()['extra'], 'value')

	def test_get_before_set_raises(self):
		service = EventLogService()
		with self.assertRaises(EventLogNotLoadedError):
			service.get_event_log()

	def test_compute_activity_frequency(self):
		service = EventLogService()
		service.set_from_dataframe(self.df.copy())
		freq = service.compute_activity_freq()
		self.assertEqual(freq.get('A'), 2)
		self.assertEqual(freq.get('B'), 1)

	def test_compute_dfg_map_and_average_time(self):
		df = pd.DataFrame(
			[
				{"case:concept:name": "C1", "concept:name": "A", "time:timestamp": "2023-01-01 00:00:00"},
				{"case:concept:name": "C1", "concept:name": "B", "time:timestamp": "2023-01-01 00:00:10"},
				{"case:concept:name": "C1", "concept:name": "C", "time:timestamp": "2023-01-01 00:00:25"},
			]
		)
		service = EventLogService()
		service.set_from_dataframe(df)
		dfg_map = service.compute_dfg_map()
		self.assertEqual(dfg_map[('A', 'B')]['count'], 1)
		self.assertEqual(dfg_map[('B', 'C')]['count'], 1)
		averages = EventLogService.compute_avg_time_from_activity(dfg_map)
		self.assertAlmostEqual(averages['A'], 10.0)
		self.assertAlmostEqual(averages['B'], 15.0)


class ProcessModelServiceTests(TestCase):
	def test_create_and_update(self):
		service = ProcessModelService()
		service.create()
		service.update(nodes=[{'id': 1}])
		self.assertEqual(service.get_process_model().nodes[0]['id'], 1)

	def test_missing_model_raises(self):
		service = ProcessModelService()
		with self.assertRaises(ProcessModelNotLoadedError):
			service.get_process_model()


class ProcessMiningServiceTests(TestCase):
	def test_create_defaults(self):
		service = ProcessMiningService()
		config = service.create(algorithm='alpha', noise_threshold=0.2)
		self.assertEqual(config.algorithm, 'alpha')
		self.assertEqual(config.noise_threshold, 0.2)

	def test_get_without_set(self):
		service = ProcessMiningService()
		with self.assertRaises(ProcessMiningNotLoadedError):
			service.get_process_mining()


class FilterConfigurationServiceTests(TestCase):
	def test_create_and_update(self):
		service = FilterConfigurationService()
		config = service.create(variant_coverage=0.8)
		self.assertEqual(config.variant_coverage, 0.8)
		service.update(event_coverage=0.5)
		self.assertEqual(service.get_config().event_coverage, 0.5)

	def test_get_without_set(self):
		service = FilterConfigurationService()
		with self.assertRaises(FilterConfigurationNotLoadedError):
			service.get_config()


class StatisticModelServiceTests(TestCase):
	def test_set_and_get_statistic_data(self):
		statistic = Statistic()
		service = StatisticModelService(statistic=statistic)
		service.set_statistic_data(
			events=[{'activity': 'A'}],
			case=[{'case_id': 'C1'}],
			variants=[{'variant': 'A-B'}],
			resources=[{'resource': 'R1'}],
		)
		self.assertEqual(service.get_statistic_data()['variants'][0]['variant'], 'A-B')

	def test_missing_statistic_raises(self):
		service = StatisticModelService()
		with self.assertRaises(StatisticNotLoadedError):
			service.get_statistic_data()


class DirectlyFollowsGraphModelServiceTests(TestCase):
	def test_create_and_update(self):
		service = DirectlyFollowsGraphModelService()
		service.create(nodes=[{'activity': 'A'}])
		service.update(edges=[{'source': 'A', 'target': 'B'}])
		graph = service.get_graph()
		self.assertEqual(graph.edges[0]['target'], 'B')

	def test_missing_graph_raises(self):
		service = DirectlyFollowsGraphModelService()
		with self.assertRaises(DirectlyFollowsGraphNotLoadedError):
			service.get_graph()
