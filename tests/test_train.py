import pytest
import numpy as np
from src.engine.etl import OctagonDataEngine
from src.engine.train import train_xgboost_predictor
from src.engine.models import FightPredictorXGB

def test_xgboost_model_training_and_calibration():
    """Enforces that XGBoost model trains on DuckDB telemetry and produces valid Brier scores (< 0.25)."""
    db_engine = OctagonDataEngine()
    db_engine.seed_historical_lightweight_dataset()
    
    predictor, brier_score = train_xgboost_predictor(db_engine)
    
    assert isinstance(predictor, FightPredictorXGB)
    assert 0.0 <= brier_score <= 0.25
    assert predictor.is_trained is True