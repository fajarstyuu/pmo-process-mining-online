import pm4py
from pathlib import Path
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from django.http import Http404, FileResponse
from process_mining.controller.ConformanceController import ConformanceController
from process_mining.controller.LoadEventLogController import LoadEventLogController
from .domain import LogParser, EventLog, EventLogFilters
from .engines import InductiveMiner, AlphaMiner, HeuristicMiner, DirectlyFollowsGraphEngine, _compute_events_statistics, _compute_case_statistics, _compute_conformance_metrics, _compute_variant_statistics, _compute_resource_statistics
from .controller.DiscoveryController import DiscoveryController
from .controller.DownloadModelController import DownloadModelController
from .controller.DiscoveryController_second import DiscoveryController as DiscoveryController2
from .controller.StatisticController import StatisticController
from .controller.FilterController import FilterController
from .controller.DownloadEventLogController import DownloadEventLogController

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
                        "case": _compute_case_statistics(event_log.df),
                        "variants": _compute_variant_statistics(event_log.df),
                        "resources": _compute_resource_statistics(event_log.df),
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
        session_id = request.data.get('session_id')
        if not session_id:
            return Response({"code": 400, "message": "No session ID provided", "data": None}, status=status.HTTP_400_BAD_REQUEST)

        noise_threshold = request.data.get('noise_threshold', 0.0)
        model_name = request.data.get('model', 'inductive')
        variants_coverage = _to_optional_float(request.data.get('variants_coverage', 1.0), default=1.0)
        events_coverage = _to_optional_float(request.data.get('events_coverage', 1.0), default=1.0)
        min_case_size = _to_optional_int(request.data.get('number_of_events_min', None), default=None)
        max_case_size = _to_optional_int(request.data.get('number_of_events_max', None), default=None)
        min_case_duration = _to_optional_float(request.data.get('min_case_duration', None), default=None)
        max_case_duration = _to_optional_float(request.data.get('max_case_duration', None), default=None)

        controller = ConformanceController()

        try:
            data = controller.apply(
                session_id=session_id,
                noise_threshold=noise_threshold,
                model_name=model_name,
                variants_coverage=variants_coverage,
                events_coverage=events_coverage,
                min_case_size=min_case_size,
                max_case_size=max_case_size,
                case_duration_min=min_case_duration,
                case_duration_max=max_case_duration,
            )
            return Response({
                "code": 200,
                "message": "Conformance check completed successfully",
                "data": data,
            }, status=status.HTTP_200_OK)
        except ValueError as exc:
            return Response({
                "code": 400,
                "message": str(exc),
                "data": None,
            }, status=status.HTTP_400_BAD_REQUEST)
        except Http404:
            return Response({
                "code": 404,
                "message": "Resource not found",
                "data": None,
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as exc:
            return Response({
                "code": 500,
                "message": f"Error during conformance check: {str(exc)}",
                "data": None,
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
class DiscoveryAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        uploaded = request.FILES.get('file')
        noise_threshold = request.data.get('noise_threshold', 0.0)
        model_name = request.data.get('model', 'inductive')
        variants_coverage = _to_optional_float(request.data.get('variants_coverage', 1.0), default=1.0)
        events_coverage = _to_optional_float(request.data.get('events_coverage', 1.0), default=1.0)
        min_case_size = _to_optional_int(request.data.get('number_of_events_min', None), default=None)
        max_case_size = _to_optional_int(request.data.get('number_of_events_max', None), default=None)
        case_duration_min = _to_optional_float(request.data.get('case_duration_min', None), default=None)
        case_duration_max = _to_optional_float(request.data.get('case_duration_max', None), default=None)

        controller = DiscoveryController()

        try:
            data = controller.apply(
                uploaded=uploaded,
                noise_threshold=noise_threshold,
                model_name=model_name,
                variants_coverage=variants_coverage,
                events_coverage=events_coverage,
                min_case_size=min_case_size,
                max_case_size=max_case_size,
                case_duration_min=case_duration_min,
                case_duration_max=case_duration_max,
            )
            return Response({
                "code": 200,
                "message": "Model discovered successfully",
                "data": data
            }, status=status.HTTP_200_OK)
        except ValueError as exc:
            return Response({
                "code": 400,
                "message": str(exc),
                "data": None
            }, status=status.HTTP_400_BAD_REQUEST)
        except Http404:
            return Response({
                "code": 404,
                "message": "Resource not found",
                "data": None
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as exc:
            print(f"Error discovering model: {str(exc)}")
            return Response({
                "code": 500,
                "message": f"Error discovering model: {str(exc)}",
                "data": None
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class ConformanceAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        uploaded = request.FILES.get('file')
        noise_threshold = request.data.get('noise_threshold', 0.0)
        model_name = request.data.get('model', 'inductive')
        variants_coverage = _to_optional_float(request.data.get('variants_coverage', 1.0), default=1.0)
        events_coverage = _to_optional_float(request.data.get('events_coverage', 1.0), default=1.0)
        min_case_duration = _to_optional_float(request.data.get('min_case_duration', 0.0), default=0.0)
        max_case_duration = _to_optional_float(request.data.get('max_case_duration', 0.0), default=0.0)

        controller = ConformanceController()

        try:
            data = controller.apply(
                uploaded=uploaded,
                noise_threshold=noise_threshold,
                model_name=model_name,
                variants_coverage=variants_coverage,
                events_coverage=events_coverage,
                min_case_size=None,
                max_case_size=None,
                case_duration_min=min_case_duration,
                case_duration_max=max_case_duration,
            )
            return Response({
                "code": 200,
                "message": "Conformance check completed successfully",
                "data": data
            }, status=status.HTTP_200_OK)
        except ValueError as exc:
            return Response({
                "code": 400,
                "message": str(exc),
                "data": None
            }, status=status.HTTP_400_BAD_REQUEST)
        except Http404:
            return Response({
                "code": 404,
                "message": "Resource not found",
                "data": None
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as exc:
            print(f"Error during conformance check: {str(exc)}")
            return Response({
                "code": 500,
                "message": f"Error during conformance check: {str(exc)}",
                "data": None
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class DownloadModelAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, format=None):
        """
        Accept a multipart/form-data upload (file) and optional form fields:
        - noise_threshold (float)
        - model (inductive|alpha|heuristic)

        Returns the PNML file of the discovered model.
        """
        session_id = request.data.get('session_id')
        if not session_id:
            return Response({"code": 400, "message": "No session ID provided", "data": None}, status=status.HTTP_400_BAD_REQUEST)
        
        noise_threshold = request.data.get('noise_threshold', 0.0)
        model_name = request.data.get('model', 'inductive')

        variants_coverage = _to_optional_float(request.data.get('variants_coverage', 1.0), default=1.0)
        events_coverage = _to_optional_float(request.data.get('events_coverage', 1.0), default=1.0)
        min_case_size = _to_optional_int(request.data.get('number_of_events_min', None), default=None)
        max_case_size = _to_optional_int(request.data.get('number_of_events_max', None), default=None)
        case_duration_min = _to_optional_float(request.data.get('case_duration_min', None), default=None)
        case_duration_max = _to_optional_float(request.data.get('case_duration_max', None), default=None)

        print(f"Received parameters: noise_threshold={noise_threshold}, model={model_name}, variants_coverage={variants_coverage}, events_coverage={events_coverage}, min_case_size={min_case_size}, max_case_size={max_case_size}, case_duration_min={case_duration_min}, case_duration_max={case_duration_max}")

        try:
            controller = DownloadModelController()
            pnml_file = controller.apply(
                session_id=session_id,
                noise_threshold=noise_threshold,
                model_name=model_name,
                variants_coverage=variants_coverage,
                events_coverage=events_coverage,
                min_case_size=min_case_size,
                max_case_size=max_case_size,
                case_duration_min=case_duration_min,
                case_duration_max=case_duration_max,
            )
            file_path: Path | None = None
            filename = "process_model.pnml"
            size_bytes = None

            if isinstance(pnml_file, dict):
                absolute_path = pnml_file.get("absolute_path")
                if absolute_path:
                    file_path = Path(absolute_path)
                filename = pnml_file.get("filename") or filename
                size_bytes = pnml_file.get("size_bytes")
            elif isinstance(pnml_file, (str, Path)):
                file_path = Path(pnml_file)

            if not file_path or not file_path.exists():
                return Response({
                    "code": 500,
                    "message": "PNML file could not be generated",
                    "data": None,
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            stream = file_path.open("rb")
            response = FileResponse(
                stream,
                as_attachment=True,
                filename=filename or file_path.name,
                content_type="application/xml",
            )

            if not size_bytes:
                try:
                    size_bytes = file_path.stat().st_size
                except OSError:
                    size_bytes = None

            response["X-Process-Message"] = "Model PNML file generated successfully"
            if size_bytes is not None:
                response["X-File-Size"] = str(size_bytes)

            return response
        except ValueError as exc:
            return Response({"code": 400, "message": str(exc), "data": None}, status=status.HTTP_400_BAD_REQUEST)
        except Http404:
            return Response({"code": 404, "message": "Resource not found", "data": None}, status=status.HTTP_404_NOT_FOUND)
        except Exception as exc:
            print(f"Error generating PNML file: {str(exc)}")
            return Response({"code": 500, "message": f"Error generating PNML file: {str(exc)}", "data": None}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class LoadEventLogAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, format=None):
        """
        Accept a multipart/form-data upload (file) and optional form fields:
        - session_id (str)

        Returns JSON: { code, message, data }
        """
        uploaded = request.FILES.get('file')
        session_id = request.data.get('session_id', None)

        controller = LoadEventLogController()

        try:
            data = controller.apply(
                uploaded=uploaded,
                session_id=session_id,
            )
            return Response({
                "code": 200,
                "message": "Event log loaded successfully",
                "data": data
            }, status=status.HTTP_200_OK)
        except ValueError as exc:
            return Response({
                "code": 400,
                "message": str(exc),
                "data": None
            }, status=status.HTTP_400_BAD_REQUEST)
        except Http404:
            return Response({
                "code": 404,
                "message": "Resource not found",
                "data": None
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as exc:
            print(f"Error loading event log: {str(exc)}")
            return Response({
                "code": 500,
                "message": f"Error loading event log: {str(exc)}",
                "data": None
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class Discovery2APIView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request):
        session_id = request.data.get('session_id', None)
        model_name = request.data.get('model_name', 'inductive')
        noise_threshold = request.data.get('noise_threshold', 0.0)

        controller = DiscoveryController2()

        try:
            data = controller.apply(
                session_id=session_id,
                model_name=model_name,
                noise_threshold=noise_threshold,
            )
            return Response({
                "code": 200,
                "message": "Model discovered successfully",
                "data": data
            }, status=status.HTTP_200_OK)
        except ValueError as exc:
            return Response({
                "code": 400,
                "message": str(exc),
                "data": None
            }, status=status.HTTP_400_BAD_REQUEST)
        except Http404:
            return Response({
                "code": 404,
                "message": "Resource not found",
                "data": None
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as exc:
            print(f"Error discovering model: {str(exc)}")
            return Response({
                "code": 500,
                "message": f"Error discovering model: {str(exc)}",
                "data": None
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class StatisticAPIView(APIView):
    def post(self, request):
        session_id = request.data.get('session_id', None)
        filtered = request.data.get('filtered', False)

        controller = StatisticController()

        try:
            data = controller.apply(
                session_id=session_id,
                filtered=filtered,
            )
            return Response({
                "code": 200,
                "message": "Statistics computed successfully",
                "data": data
            }, status=status.HTTP_200_OK)
        except ValueError as exc:
            return Response({
                "code": 400,
                "message": str(exc),
                "data": None
            }, status=status.HTTP_400_BAD_REQUEST)
        except Http404:
            return Response({
                "code": 404,
                "message": "Resource not found",
                "data": None
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as exc:
            print(f"Error computing statistics: {str(exc)}")
            return Response({
                "code": 500,
                "message": f"Error computing statistics: {str(exc)}",
                "data": None
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class FilterAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, format=None):
        """
        Accept a multipart/form-data upload (file) and optional form fields:
        - session_id (str)
        - variants_coverage (float)
        - events_coverage (float)
        - min_case_size (int)
        - max_case_size (int)
        - case_duration_min (float)
        - case_duration_max (float)

        Returns JSON: { code, message, data }
        """

        session_id = request.data.get('session_id', None)
        variants_coverage = _to_optional_float(request.data.get('variants_coverage', 1.0), default=1.0)
        events_coverage = _to_optional_float(request.data.get('events_coverage', 1.0), default=1.0)
        min_case_size = _to_optional_int(request.data.get('min_case_size', None), default=None)
        max_case_size = _to_optional_int(request.data.get('max_case_size', None), default=None)
        case_duration_min = _to_optional_float(request.data.get('case_duration_min', None), default=None)
        case_duration_max = _to_optional_float(request.data.get('case_duration_max', None), default=None)

        print(f"Received parameters: session_id={session_id}, variants_coverage={variants_coverage}, events_coverage={events_coverage}, min_case_size={min_case_size}, max_case_size={max_case_size}, case_duration_min={case_duration_min}, case_duration_max={case_duration_max}")

        controller = FilterController()

        try:
            data = controller.apply(
                session_id=session_id,
                variants_coverage=variants_coverage,
                events_coverage=events_coverage,
                min_case_size=min_case_size,
                max_case_size=max_case_size,
                case_duration_min=case_duration_min,
                case_duration_max=case_duration_max,
            )
            return Response({
                "code": 200,
                "message": "Event log filtered successfully",
                "data": data
            }, status=status.HTTP_200_OK)
        except ValueError as exc:
            return Response({
                "code": 400,
                "message": str(exc),
                "data": None
            }, status=status.HTTP_400_BAD_REQUEST)
        except Http404:
            return Response({
                "code": 404,
                "message": "Resource not found",
                "data": None
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as exc:
            print(f"Error filtering event log: {str(exc)}")
            return Response({
                "code": 500,
                "message": f"Error filtering event log: {str(exc)}",
                "data": None
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class DownloadEventLogAPIView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, format=None):
        """
        Accept a multipart/form-data upload (file) and optional form fields:
        - file_type (str): 'xes' or 'csv'

        Returns the converted event log file.
        """
        session_id = request.data.get('session_id')
        if not session_id:
            return Response({"code": 400, "message": "No session ID provided", "data": None}, status=status.HTTP_400_BAD_REQUEST)

        controller = DownloadEventLogController()

        try:
            file_info = controller.apply(
                session_id=session_id,
            )

            target_path = Path(file_info.get("absolute_path"))
            if not target_path.exists():
                return Response({
                    "code": 500,
                    "message": "Converted file could not be generated",
                    "data": None,
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            stream = target_path.open("rb")
            response = FileResponse(
                stream,
                as_attachment=True,
                filename=file_info.get("filename", target_path.name),
                content_type="application/octet-stream",
            )

            response["X-Process-Message"] = "Event log converted successfully"
            response["X-File-Size"] = str(file_info.get("size_bytes", 0))
            response["X-Source-Log"] = file_info.get("source_path", "")

            return response
        except ValueError as exc:
            return Response({
                "code": 400,
                "message": str(exc),
                "data": None
            }, status=status.HTTP_400_BAD_REQUEST)
        except Http404:
            return Response({
                "code": 404,
                "message": "Resource not found",
                "data": None
            }, status=status.HTTP_404_NOT_FOUND)
        except Exception as exc:
            print(f"Error converting event log: {str(exc)}")
            return Response({
                "code": 500,
                "message": f"Error converting event log: {str(exc)}",
                "data": None
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

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