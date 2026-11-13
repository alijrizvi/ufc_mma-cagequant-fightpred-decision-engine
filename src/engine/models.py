from typing import Any, Dict, Tuple

import numpy as np
import xgboost as xgb

from src.engine.stats import FighterCombatProfile


class FightPredictorXGB:
    """Predictive Machine Learning engine powered by XGBoost for 10-second interval probabilities."""

    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=42
        )
        self.is_trained = False

    def extract_features(
        self, 
        interval_state: Dict[str, Any], 
        profile_a: FighterCombatProfile, 
        profile_b: FighterCombatProfile
    ) -> np.ndarray:
        """Extracts a normalized 2D feature vector from the 10-second interval state."""
        hp_a = float(interval_state.get("fighter_a_hp", 100.0))
        hp_b = float(interval_state.get("fighter_b_hp", 100.0))
        stamina_a = float(interval_state.get("fighter_a_stamina", 100.0))
        stamina_b = float(interval_state.get("fighter_b_stamina", 100.0))
        
        dmg_a = float(interval_state.get("damage_dealt_a", 0.0))
        dmg_b = float(interval_state.get("damage_dealt_b", 0.0))
        
        damage_diff = dmg_a - dmg_b
        power_diff = profile_a.punch_power - profile_b.punch_power

        return np.array([[
            float(interval_state.get("round_num", 1)),
            hp_a,
            hp_b,
            stamina_a,
            stamina_b,
            damage_diff,
            power_diff
        ]], dtype=np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fits the XGBoost classifier on feature matrix X and target binary label y."""
        self.model.fit(X, y)
        self.is_trained = True

    def predict_interval_win_probability(
        self, 
        interval_state: Dict[str, Any], 
        profile_a: FighterCombatProfile, 
        profile_b: FighterCombatProfile
    ) -> Tuple[float, float]:
        """Calculates calibrated win probabilities P_win(A) and P_win(B) for the current interval."""
        features = self.extract_features(interval_state, profile_a, profile_b)
        
        if self.is_trained:
            probs = self.model.predict_proba(features)[0]
            p_win_a = float(probs[1]) if len(probs) > 1 else float(probs[0])
            p_win_b = float(1.0 - p_win_a)
        else:
            # Fallback prior prior to explicit model fitting
            hp_a, hp_b = features[0, 1], features[0, 2]
            stamina_a, stamina_b = features[0, 3], features[0, 4]
            score_a = (hp_a * 0.6) + (stamina_a * 0.4)
            score_b = (hp_b * 0.6) + (stamina_b * 0.4)
            logit = (score_a - score_b) / 20.0
            p_win_a = float(1.0 / (1.0 + np.exp(-logit)))
            p_win_b = float(1.0 - p_win_a)

        return round(p_win_a, 4), round(p_win_b, 4)