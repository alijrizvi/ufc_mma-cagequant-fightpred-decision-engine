import numpy as np
import pytest

# Importing FightPredictorXGB before implementing it to force the Red TDD state
from src.engine.models import FightPredictorXGB
from src.engine.stats import FighterCombatProfile


def test_feature_vector_extraction():
    """Enforces that a 10-second interval state maps into a valid non-null numeric feature vector."""
    predictor = FightPredictorXGB()
    
    profile_a = FighterCombatProfile(name="Islam Makhachev", punch_power=25.0)
    profile_b = FighterCombatProfile(name="Carlos Prates", punch_power=20.0)
    
    interval_state = {
        "round_num": 1,
        "time_seconds": 60,
        "fighter_a_hp": 85.0,
        "fighter_b_hp": 70.0,
        "fighter_a_stamina": 90.0,
        "fighter_b_stamina": 80.0,
        "damage_dealt_a": 15.0,
        "damage_dealt_b": 5.0
    }
    
    features = predictor.extract_features(interval_state, profile_a, profile_b)
    
    assert isinstance(features, np.ndarray)
    assert features.ndim == 2
    assert features.shape[1] >= 6  # At least 6 domain features extracted
    assert not np.isnan(features).any()

def test_win_probability_bounds():
    """Enforces that predicted win probabilities satisfy P in [0.0, 1.0] and sum to 1.0."""
    predictor = FightPredictorXGB()
    
    profile_a = FighterCombatProfile(name="Islam Makhachev")
    profile_b = FighterCombatProfile(name="Carlos Prates")
    
    interval_state = {
        "round_num": 2,
        "time_seconds": 120,
        "fighter_a_hp": 90.0,
        "fighter_b_hp": 40.0,
        "fighter_a_stamina": 85.0,
        "fighter_b_stamina": 50.0,
        "damage_dealt_a": 25.0,
        "damage_dealt_b": 5.0
    }
    
    p_win_a, p_win_b = predictor.predict_interval_win_probability(interval_state, profile_a, profile_b)
    
    assert 0.0 <= p_win_a <= 1.0
    assert 0.0 <= p_win_b <= 1.0
    assert pytest.approx(p_win_a + p_win_b, rel=1e-3) == 1.0