from pathlib import Path

from django.conf import settings
from django.utils import timezone
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter


class DownloadModelService:
    def __init__(self, output_dir: str | Path | None = None):
        base_dir = Path(output_dir) if output_dir else Path(settings.BASE_DIR)
        base_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = base_dir

    def apply(self, petri_net, initial_marking, final_marking, filename: str | None = None):
        timestamp = timezone.now().strftime("%Y%m%d%H%M%S")
        safe_filename = filename or f"process_model_{timestamp}.pnml"
        file_path = self.output_dir / safe_filename

        pnml_exporter.apply(petri_net, initial_marking, str(file_path), final_marking)

        return {
            "filename": safe_filename,
            "absolute_path": str(file_path.resolve()),
            "size_bytes": file_path.stat().st_size if file_path.exists() else 0,
        }