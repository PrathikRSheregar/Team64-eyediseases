import json
from fastapi import APIRouter, HTTPException, status
from ml.config import REPORTS_DIR

router = APIRouter(prefix="/dataset", tags=["Dataset"])


@router.get("/statistics", summary="Dataset size, class distribution, and split statistics")
def dataset_statistics():
    stats_path = REPORTS_DIR / "dataset_statistics.json"
    if not stats_path.exists():
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail=(
                "No dataset statistics found yet. Run "
                "'python -m ml.data.analyze_dataset' after placing your "
                "dataset under dataset/<class_name>/."
            ),
        )
    return json.loads(stats_path.read_text())
