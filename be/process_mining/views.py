import pm4py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from django.http import Http404

from .domain import LogParser, EventLog, EventLogFilters
from .engines import InductiveMiner, AlphaMiner, HeuristicMiner, DirectlyFollowsGraphEngine, _compute_events_statistics, _compute_case_statistics, _compute_conformance_metrics


class IndexView(APIView):
    def get(self, request):
        return Response({"message": "Process Mining API (Django)"})


class DiscoverModelAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, format=None):
        """
        Accept a multipart/form-data upload (file) and optional form fields:
        - noise_threshold (float)
        - model (inductive|alpha|heuristic)

        Returns JSON: { code, message, data }
        """
        uploaded = request.FILES.get('file')
        if not uploaded:
            return Response({"code": 400, "message": "No file provided", "data": None}, status=status.HTTP_400_BAD_REQUEST)

        # Raw inputs may be empty strings; normalize and parse safely
        noise_threshold = request.data.get('noise_threshold', 0.0)
        model_name = request.data.get('model', 'inductive')

        def _to_optional_float(val, default=None):
            if val is None:
                return default
            if isinstance(val, str) and val.strip() == "":
                return default
            try:
                return float(val)
            except Exception:
                return default

        def _to_optional_int(val, default=None):
            if val is None:
                return default
            if isinstance(val, str) and val.strip() == "":
                return default
            try:
                return int(float(val))  # allow "3.0" style inputs
            except Exception:
                return default

        variants_coverage = _to_optional_float(request.data.get('variants_coverage', 1.0), default=1.0)
        events_coverage = _to_optional_float(request.data.get('events_coverage', 1.0), default=1.0)
        min_case_size = _to_optional_int(request.data.get('number_of_events_min', None), default=None)
        max_case_size = _to_optional_int(request.data.get('number_of_events_max', None), default=None)
        case_duration_min = _to_optional_float(request.data.get('case_duration_min', None), default=None)
        case_duration_max = _to_optional_float(request.data.get('case_duration_max', None), default=None)

        print(f"Received parameters: noise_threshold={noise_threshold}, model={model_name}, variants_coverage={variants_coverage}, events_coverage={events_coverage}, min_case_size={min_case_size}, max_case_size={max_case_size}, case_duration_min={case_duration_min}, case_duration_max={case_duration_max}")

        try:
            noise_threshold = float(noise_threshold)
        except Exception:
            noise_threshold = 0.0

        try:
            # print(f"DiscoverModelAPIView: model={model_name}, noise_threshold={noise_threshold}, variants_coverage={variants_coverage}, events_coverage={events_coverage}")
            # Parse file into EventLog
            parser = LogParser()
            event_log = parser.parse(uploaded)

            # Apply filters
            event_log = EventLogFilters.variants_coverage(event_log, coverage_target=variants_coverage or 1.0)
            event_log = EventLogFilters.events_coverage(event_log, coverage_target=events_coverage or 1.0)

            # Apply case size filter only when both bounds are provided and valid
            if min_case_size is not None and max_case_size is not None:
                event_log = EventLogFilters.case_size(event_log, min_size=min_case_size, max_size=max_case_size)

            # Apply case duration filter only when both bounds are provided and valid
            if case_duration_min is not None and case_duration_max is not None:
                event_log = EventLogFilters.case_duration(event_log, min_performance=case_duration_min, max_performance=case_duration_max)

            # Choose engine
            model_key = str(model_name).lower() if model_name else 'inductive'
            if model_key == 'alpha':
                engine = AlphaMiner()
            elif model_key == 'heuristic':
                engine = HeuristicMiner()
            else:
                engine = InductiveMiner()

            process_model = engine.discover(event_log, noise_threshold=noise_threshold)
            dfg_model = DirectlyFollowsGraphEngine().discover(event_log, include_performance=True)

            return Response({
                "code": 200,
                "message": "Model discovered successfully",
                "data": {
                    "petri_net": process_model.to_cytoscape(),
                    "dfg": dfg_model.to_cytoscape(),
                    "model_statistics": {
                        "events": _compute_events_statistics(event_log.df),
                        "case": _compute_case_statistics(event_log.df)
                    }
                },
            }, status=status.HTTP_200_OK)

        except Http404:
            return Response({"code": 404, "message": "Resource not found", "data": None}, status=status.HTTP_404_NOT_FOUND)
        except Exception as exc:
            # Catch-all to return standardized JSON
            print(f"Error discovering model: {str(exc)}")
            return Response({"code": 500, "message": f"Error discovering model: {str(exc)}", "data": None}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
class ConformanceCheckAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, format=None):
        """
        Accept a multipart/form-data upload (file) and a model file:
        - log_file (file)
        - model_file (file)

        Returns JSON: { code, message, data }
        """
        uploaded = request.FILES.get('file')
        if not uploaded:
            return Response({"code": 400, "message": "No file provided", "data": None}, status=status.HTTP_400_BAD_REQUEST)
        
        noise_threshold = request.data.get('noise_threshold', 0.0)
        model_name = request.data.get('model', 'inductive')
        variants_coverage = float(request.data.get('variants_coverage', 1.0))
        events_coverage = float(request.data.get('events_coverage', 1.0))
        min_case_duration = float(request.data.get('min_case_duration', 0.0))
        max_case_duration = float(request.data.get('max_case_duration', 0.0))

        try:
            parser = LogParser()
            event_log = parser.parse(uploaded)

            # make copy of event_log for conformance checking
            event_log_for_conformance = EventLog(event_log.df.copy())

            # Apply filters
            event_log_for_conformance = EventLogFilters.variants_coverage(event_log_for_conformance, coverage_target=variants_coverage)
            event_log_for_conformance = EventLogFilters.events_coverage(event_log_for_conformance, coverage_target=events_coverage)
            event_log_for_conformance = EventLogFilters.case_duration(event_log_for_conformance, min_performance=min_case_duration, max_performance=max_case_duration)

            # Choose engine
            model_key = str(model_name).lower() if model_name else 'inductive'
            if model_key == 'alpha':
                net, im, fm = pm4py.discover_petri_net_alpha(event_log_for_conformance.df, noise_threshold=noise_threshold)
            elif model_key == 'heuristic':
                net, im, fm = pm4py.discover_petri_net_heuristics(event_log_for_conformance.df)
            elif model_key == 'inductive':
                net, im, fm = pm4py.discover_petri_net_inductive(event_log_for_conformance.df, noise_threshold=noise_threshold)
            else:
                return Response({"code": 400, "message": f"Unknown model type: {model_name}", "data": None}, status=status.HTTP_400_BAD_REQUEST)


            conformance_results = _compute_conformance_metrics(event_log_for_conformance, net, im, fm)
            return Response({
                "code": 200,
                "message": "Conformance check completed successfully",
                "data": conformance_results,
            }, status=status.HTTP_200_OK)

        except Http404:
            return Response({"code": 404, "message": "Resource not found", "data": None}, status=status.HTTP_404_NOT_FOUND)
        except Exception as exc:
            return Response({"code": 500, "message": f"Error during conformance check: {str(exc)}", "data": None}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)