from src.engine.etl import OctagonDataEngine
from src.engine.train import CalibratedModel, ModelTrainer, train_xgboost_predictor


def test_xgboost_model_training_and_calibration(tmp_path):
    """The default synthetic pipeline trains a calibrated model with Brier below the coin-flip 0.25."""
    db_engine = OctagonDataEngine()
    db_engine.seed_historical_lightweight_dataset()

    predictor, brier_score = train_xgboost_predictor(db_engine, model_dir=str(tmp_path))

    assert isinstance(predictor, CalibratedModel)
    assert 0.0 <= brier_score < 0.25


def test_synthetic_pipeline_has_positive_skill(tmp_path):
    """FR-03: BSS > 0 on the (disclosed) synthetic frame. Proves plumbing, not real-world skill."""
    db_engine = OctagonDataEngine()
    db_engine.seed_historical_lightweight_dataset()
    trainer = ModelTrainer(model_dir=str(tmp_path))
    bss, _ = trainer.train_and_evaluate(db_engine.get_training_frame())
    assert bss > 0.0
