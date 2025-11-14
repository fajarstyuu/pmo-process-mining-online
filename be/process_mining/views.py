from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from django.http import Http404

from .domain import LogParser, EventLog
from .engines import InductiveMiner, AlphaMiner, HeuristicMiner


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

        noise_threshold = request.data.get('noise_threshold', 0.0)
        model_name = request.data.get('model', 'inductive')

        try:
            noise_threshold = float(noise_threshold)
        except Exception:
            noise_threshold = 0.0

        try:
            # Parse file into EventLog
            parser = LogParser()
            event_log = parser.parse(uploaded)

            # Choose engine
            model_key = str(model_name).lower() if model_name else 'inductive'
            if model_key == 'alpha':
                engine = AlphaMiner()
            elif model_key == 'heuristic':
                engine = HeuristicMiner()
            else:
                engine = InductiveMiner()

            process_model = engine.discover(event_log, noise_threshold=noise_threshold)

            return Response({
                "code": 200,
                "message": "Model discovered successfully",
                "data": process_model.to_cytoscape()
            }, status=status.HTTP_200_OK)

        except Http404:
            return Response({"code": 404, "message": "Resource not found", "data": None}, status=status.HTTP_404_NOT_FOUND)
        except Exception as exc:
            # Catch-all to return standardized JSON
            return Response({"code": 500, "message": f"Error discovering model: {str(exc)}", "data": None}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)