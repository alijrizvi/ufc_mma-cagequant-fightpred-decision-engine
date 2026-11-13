import pytest

from src.engine.stats import CombatPillarEngine, FighterCombatProfile


def test_strike_damage_mitigation_calculation():
    """Enforces that strong defensive mitigation reduces raw strike damage correctly."""
    attacker = FighterCombatProfile(
        name="Conor McGregor",
        punch_power=30.0,       # Base strike power
        strike_hit_pct=0.55,
        stamina_decay_rate=0.05
    )
    defender = FighterCombatProfile(
        name="Dustin Poirier",
        arm_block_pct=0.40,     # Mitigates 40% of incoming damage
        durability_index=1.0
    )
    
    engine = CombatPillarEngine()
    net_damage = engine.calculate_interval_strike_damage(attacker, defender, attacker_stamina=100.0)
    
    # Expected: 30.0 * (1.0 - 0.40) * 1.0 = 18.0 HP
    assert net_damage == pytest.approx(18.0, rel=1e-2)

def test_stamina_fatigue_penalty():
    """Enforces that depleted stamina reduces outgoing strike power proportionally."""
    attacker = FighterCombatProfile(name="Tired Fighter", punch_power=30.0)
    defender = FighterCombatProfile(name="Fresh Fighter", arm_block_pct=0.0)
    
    engine = CombatPillarEngine()
    # At 50% stamina, damage output should be halved
    fatigued_damage = engine.calculate_interval_strike_damage(attacker, defender, attacker_stamina=50.0)
    
    assert fatigued_damage == pytest.approx(15.0, rel=1e-2)