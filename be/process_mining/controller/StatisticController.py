from process_mining.domain import LogParser
from process_mining.model.EventLog import EventLog as StoredEventLog
from process_mining.model.Statistic import Statistic
from process_mining.service.EventLogService import EventLogService
from process_mining.service.StatisticService import StatisticService
from process_mining.service.StatisticModelService import StatisticModelService
    

class StatisticController:
    def apply(self, session_id=None, filtered=False):
        if not session_id:
            raise ValueError("No session ID provided")

        event_log_service = EventLogService()
        stored_event_log = StoredEventLog()
        event_log_service.set_event_log(event_log=stored_event_log)
        event_log_service.set_id(str(session_id))

        log_path = event_log_service.get_event_logs(filtered=filtered)
        parser = LogParser()
        with open(log_path, "rb") as stored_file:
            event_log = parser.parse(stored_file)

        print(f"Computing statistics for event log for session_id={session_id} using file {log_path}")
        statistic_model = Statistic()
        statistic_service = StatisticService()
        statistic_model_service = StatisticModelService(statistic_model)
        statistic_model_service.set_statistic_data(
            events=statistic_service.compute_events_statistics(event_log.df),
            case=statistic_service.compute_case_statistics(event_log.df),
            variants=statistic_service.compute_variant_statistics(event_log.df),
            resources=statistic_service.compute_resource_statistics(event_log.df),
            general=statistic_service.compute_model_statistics(
                petri_net=None,
                noise_threshold=0.0,
                event_log_df=event_log.df
            )
        )

        return statistic_model_service.get_statistic_data()