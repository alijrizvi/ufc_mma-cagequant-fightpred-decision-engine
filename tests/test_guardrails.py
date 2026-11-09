import pytest
import time
from src.engine.simulation import OctagonSimulationEngine, FighterState
from src.engine.stats import FighterCombatProfile

def test_fighter_initial_hp_boundary():
    """Enforces that both fighters initialize at exactly 100.0 HP."""
    profile_a = FighterCombatProfile(name="Islam Makhachev")
    profile_b = FighterCombatProfile(name="Carlos Prates")
    
    fighter_a = FighterState(name=profile_a.name, profile=profile_a, base_hp=100.0)
    fighter_b = FighterState(name=profile_b.name, profile=profile_b, base_hp=100.0)
    
    assert fighter_a.hp == 100.0
    assert fighter_b.hp == 100.0

def test_interval_hp_non_negative():
    """Enforces that damage calculation never drops HP below zero."""
    profile = FighterCombatProfile(name="Test Fighter")
    engine = OctagonSimulationEngine()
    fighter = FighterState(name=profile.name, profile=profile, base_hp=10.0)
    
    updated_fighter = engine.apply_interval_damage(fighter, raw_damage=50.0, mitigation=0.0)
    
    assert updated_fighter.hp >= 0.0
    assert updated_fighter.hp == 0.0

def test_simulation_victory_method_generation():
    """Enforces that every fight simulation outputs a valid method of victory string."""
    profile_a = FighterCombatProfile(name="Dustin Poirier", punch_power=35.0)
    profile_b = FighterCombatProfile(name="Justin Gaethje", punch_power=10.0)
    
    engine = OctagonSimulationEngine()
    result = engine.run_full_fight(profile_a, profile_b, max_rounds=5)
    
    assert result.method_of_victory != ""
    assert result.winner in [profile_a.name, profile_b.name, "Draw"]
    assert result.finish_round >= 1