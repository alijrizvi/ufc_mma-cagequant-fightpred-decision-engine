import numpy as np
from typing import Tuple
from sklearn.metrics import brier_score_loss
from src.engine.etl import OctagonDataEngine
from src.engine.models import FightPredictorXGB

def build_training_dataset(db_engine: OctagonDataEngine) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts 10-second interval states from DuckDB telemetry into X (features) and y (binary outcomes)."""
    db_engine.seed_historical_lightweight_dataset()
    
    df_telemetry = db_engine.conn.execute("""
        SELECT interval_id, round_num, fighter_name, strikes_landed, damage_dealt, clinch_seconds, ground_seconds
        FROM interval_telemetry
    """).df()

    X_list, y_list = [], []
    profile_a = db_engine.load_fighter_profile("Islam Makhachev")
    profile_b = db_engine.load_fighter_profile("Carlos Prates")
    predictor = FightPredictorXGB()

    for _, row in df_telemetry.iterrows():
        # Scenario 1: Red Corner Advantage State
        state_a_lead = {
            "round_num": int(row["round_num"]),
            "time_seconds": int(row["interval_id"]) * 10,
            "fighter_a_hp": 85.0,
            "fighter_b_hp": 60.0,
            "fighter_a_stamina": 90.0,
            "fighter_b_stamina": 70.0,
            "damage_dealt_a": float(row["damage_dealt"]),
            "damage_dealt_b": 5.0
        }
        feat_a = predictor.extract_features(state_a_lead, profile_a, profile_b)[0]
        X_list.append(feat_a)
        y_list.append(1)

        # Scenario 2: Blue Corner Advantage State
        state_b_lead = {
            "round_num": int(row["round_num"]),
            "time_seconds": int(row["interval_id"]) * 10,
            "fighter_a_hp": 55.0,
            "fighter_b_hp": 90.0,
            "fighter_a_stamina": 65.0,
            "fighter_b_stamina": 95.0,
            "damage_dealt_a": 4.0,
            "damage_dealt_b": float(row["damage_dealt"])
        }
        feat_b = predictor.extract_features(state_b_lead, profile_a, profile_b)[0]
        X_list.append(feat_b)
        y_list.append(0)

    return np.array(X_list), np.array(y_list)

def train_xgboost_predictor(db_engine: OctagonDataEngine) -> Tuple[FightPredictorXGB, float]:
    """Fits XGBoost classifier on interval telemetry matrix and returns (predictor, brier_score)."""
    X, y = build_training_dataset(db_engine)
    
    predictor = FightPredictorXGB()
    predictor.fit(X, y)

    # Compute Brier score loss for probability calibration quality
    y_probs = predictor.model.predict_proba(X)[:, 1]
    brier = float(brier_score_loss(y, y_probs))

    return predictor, round(brier, 4)