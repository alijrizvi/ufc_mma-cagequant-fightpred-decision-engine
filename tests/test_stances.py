import pytest
from src.engine.stats import FighterCombatProfile
from src.engine.stances import TacticalStance, TacticalStanceEngine

def test_stance_attribute_modifiers():
    """Enforces that stance multipliers correctly alter base combat parameters."""
    profile = FighterCombatProfile(name="Justin Gaethje", punch_power=30.0, arm_block_pct=0.40, stamina_decay_rate=0.04)
    stance_engine = TacticalStanceEngine()

    modified = stance_engine.apply_stance_modifiers(profile, TacticalStance.POWER_OVERDRIVE)

    assert modified.punch_power == pytest.approx(30.0 * 1.40)
    assert modified.stamina_decay_rate == pytest.approx(0.04 * 2.50)
    assert modified.arm_block_pct == pytest.approx(0.40 * 0.75)

def test_auto_stance_selection_rules():
    """Enforces that ASCENDED_SSJ2 requires high HP and Stamina baselines."""
    stance_engine = TacticalStanceEngine()

    # High HP & Stamina -> SSJ2
    stance_high = stance_engine.determine_optimal_stance(hp=85.0, stamina=80.0, preferred_mode="aggressive")
    assert stance_high == TacticalStance.ASCENDED_SSJ2

    # Depleted Stamina -> Mastered Flow to preserve energy
    stance_low = stance_engine.determine_optimal_stance(hp=40.0, stamina=25.0, preferred_mode="aggressive")
    assert stance_low == TacticalStance.MASTERED_FLOW