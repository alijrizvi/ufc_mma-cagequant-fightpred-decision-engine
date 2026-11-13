"""
Model Training Pipeline: Calibrated XGBoost with Group-Aware Splitting.

Public interface:
- CalibratedModel
- ModelTrainer
- train_xgboost_predictor
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GroupShuffleSplit


class CalibratedModel:
    """Wraps a fitted base classifier and an isotonic calibrator."""

    def __init__(self, base_estimator: Any, calibrator: IsotonicRegression, feature_names: List[str]):
        self.base_estimator = base_estimator
        self.calibrator = calibrator
        self.feature_names = feature_names

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X_mat = X.reindex(columns=self.feature_names, fill_value=0.0)
        else:
            X_mat = X
        raw_probs = self.base_estimator.predict_proba(X_mat)[:, 1]
        cal_probs = np.clip(self.calibrator.predict(raw_probs), 0.0, 1.0)
        return np.column_stack([1.0 - cal_probs, cal_probs])

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_interval_win_probability(
        self,
        interval_state: Dict[str, Any],
        profile_a: Any,
        profile_b: Any,
    ) -> Tuple[float, float]:
        """Calculates calibrated win probabilities P_win(A) and P_win(B) for a simulation interval."""
        hp_a = float(interval_state.get("fighter_a_hp", 100.0))
        hp_b = float(interval_state.get("fighter_b_hp", 100.0))
        stam_a = float(interval_state.get("fighter_a_stamina", 100.0))
        stam_b = float(interval_state.get("fighter_b_stamina", 100.0))
        dmg_a = float(interval_state.get("damage_dealt_a", 0.0))
        dmg_b = float(interval_state.get("damage_dealt_b", 0.0))
        row = pd.DataFrame(
            [
                {
                    "round_num": float(interval_state.get("round_num", 1)),
                    "fighter_a_hp": hp_a,
                    "fighter_b_hp": hp_b,
                    "fighter_a_stamina": stam_a,
                    "fighter_b_stamina": stam_b,
                    "damage_diff": dmg_a - dmg_b,
                    "power_diff": float(profile_a.punch_power - profile_b.punch_power),
                    "hp_delta": hp_a - hp_b,
                    "stamina_delta": stam_a - stam_b,
                }
            ]
        )
        p_win_a = float(self.predict_proba(row)[0, 1])
        return round(p_win_a, 4), round(1.0 - p_win_a, 4)


class ModelTrainer:
    def __init__(self, model_dir: str = "models/"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: Dict[str, float] = {}

        self.base_estimator = xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
        )

    def train_and_evaluate(self, df: pd.DataFrame) -> Tuple[float, CalibratedModel]:
        """Executes group-split training, isotonic calibration, and BSS evaluation."""
        X = df.drop(columns=["bout_winner"])
        y = df["bout_winner"]
        groups = df["fight_id"]

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(X, y, groups=groups))

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        X_train_features = X_train.drop(columns=["fight_id"], errors="ignore")
        X_test_features = X_test.drop(columns=["fight_id"], errors="ignore")
        feature_names = list(X_train_features.columns)

        val_gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        t_idx, v_idx = next(val_gss.split(X_train_features, y_train, groups=X_train["fight_id"]))

        self.base_estimator.fit(X_train_features.iloc[t_idx], y_train.iloc[t_idx])
        val_raw = self.base_estimator.predict_proba(X_train_features.iloc[v_idx])[:, 1]

        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(val_raw, y_train.iloc[v_idx])

        model = CalibratedModel(
            base_estimator=self.base_estimator,
            calibrator=calibrator,
            feature_names=feature_names,
        )

        calibrated_probs = model.predict_proba(X_test_features)[:, 1]
        bs_model = float(brier_score_loss(y_test, calibrated_probs))
        base_rate = float(y_train.mean())
        bs_baseline = float(brier_score_loss(y_test, [base_rate] * len(y_test)))
        bss = float(1.0 - (bs_model / bs_baseline)) if bs_baseline > 0 else 0.0

        self.metrics = {
            "brier_model": bs_model,
            "brier_baseline": bs_baseline,
            "base_rate": base_rate,
            "bss": bss,
        }

        if bss > 0.0:
            joblib.dump(model, self.model_dir / "calibrated_xgb.pkl")

        return bss, model


def _resolve_frame(source: Optional[Union[pd.DataFrame, Any]]) -> pd.DataFrame:
    """Accepts None (synthetic demo), a DataFrame, or an engine with get_training_frame()."""
    if source is None:
        from src.engine.etl import OctagonDataEngine  # local import avoids a module cycle

        return OctagonDataEngine().get_training_frame()
    if isinstance(source, pd.DataFrame):
        return source
    if hasattr(source, "get_training_frame"):
        return source.get_training_frame()
    raise TypeError(f"Unsupported training data source: {type(source)!r}")


def train_xgboost_predictor(
    source: Optional[Union[pd.DataFrame, Any]] = None,
    model_dir: str = "models/",
) -> Tuple[CalibratedModel, float]:
    """Trains a calibrated XGBoost predictor and returns (model, brier_score)."""
    trainer = ModelTrainer(model_dir=model_dir)
    _, model = trainer.train_and_evaluate(_resolve_frame(source))
    if trainer.metrics["bss"] <= 0.0:
        warnings.warn(
            f"Model has no skill over the base rate (BSS={trainer.metrics['bss']:.3f}); "
            "do not trust its probabilities. Add data or revisit features.",
            stacklevel=2,
        )
    return model, trainer.metrics["brier_model"]
