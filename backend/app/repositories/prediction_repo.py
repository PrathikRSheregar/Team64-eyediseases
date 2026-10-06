from sqlalchemy.orm import Session, joinedload
from app.models.prediction import Prediction, PredictionProbability


def create_prediction(
    db: Session, user_id: str, image_path: str, predicted_class: str,
    confidence: float, model_name: str, probabilities: dict[str, float],
    gradcam_path: str | None = None, model_version: str = "v1",
) -> Prediction:
    pred = Prediction(
        user_id=user_id, image_path=image_path, predicted_class=predicted_class,
        confidence=confidence, model_name=model_name, model_version=model_version,
        gradcam_path=gradcam_path,
    )
    db.add(pred)
    db.flush()  # get pred.id before commit

    probs = PredictionProbability(
        prediction_id=pred.id,
        cataract_probability=probabilities.get("Cataract", 0.0),
        diabetic_retinopathy_probability=probabilities.get("Diabetic Retinopathy", 0.0),
        glaucoma_probability=probabilities.get("Glaucoma", 0.0),
        normal_probability=probabilities.get("Normal", 0.0),
    )
    db.add(probs)
    db.commit()
    db.refresh(pred)
    return pred


def get_by_id(db: Session, prediction_id: str) -> Prediction | None:
    return (
        db.query(Prediction)
        .options(joinedload(Prediction.probabilities))
        .filter(Prediction.id == prediction_id)
        .first()
    )


def list_for_user(
    db: Session, user_id: str, disease: str | None = None,
    model_name: str | None = None, limit: int = 50, offset: int = 0,
) -> list[Prediction]:
    q = db.query(Prediction).filter(Prediction.user_id == user_id)
    if disease:
        q = q.filter(Prediction.predicted_class == disease)
    if model_name:
        q = q.filter(Prediction.model_name == model_name)
    return q.order_by(Prediction.created_at.desc()).offset(offset).limit(limit).all()


def list_all(db: Session, limit: int = 100, offset: int = 0) -> list[Prediction]:
    return (
        db.query(Prediction)
        .order_by(Prediction.created_at.desc())
        .offset(offset).limit(limit).all()
    )


def count_all(db: Session) -> int:
    return db.query(Prediction).count()


def count_by_class(db: Session) -> dict[str, int]:
    from sqlalchemy import func
    rows = db.query(Prediction.predicted_class, func.count(Prediction.id)).group_by(Prediction.predicted_class).all()
    return {r[0]: r[1] for r in rows}
