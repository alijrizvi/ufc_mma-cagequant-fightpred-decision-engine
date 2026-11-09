import pytest
from src.engine.etl import OctagonDataEngine
from src.engine.stats import FighterCombatProfile

# We enforce that our DuckDB engine can load pre-populated Fighter combat profiles dynamically from 
# Relational Tables rather than requiring manual instantiation in Python

def test_load_fighter_profile_from_duckdb():
    """Enforces that fighter profiles can be queried dynamically from DuckDB telemetry tables."""
    db_engine = OctagonDataEngine()
    db_engine.seed_historical_lightweight_dataset()
    
    profile = db_engine.load_fighter_profile("Islam Makhachev")
    
    assert isinstance(profile, FighterCombatProfile)
    assert profile.name == "Islam Makhachev"
    assert profile.punch_power > 0.0
    assert 0.0 <= profile.arm_block_pct <= 1.0

def test_computed_advanced_metrics_query():
    """Enforces that DuckDB calculates xCT, SDR, and SVI features accurately."""
    db_engine = OctagonDataEngine()
    db_engine.seed_historical_lightweight_dataset()
    
    metrics = db_engine.get_advanced_feature_metrics("Carlos Prates")
    
    assert "xct" in metrics
    assert "stamina_drain_ratio" in metrics
    assert "strike_volatility_idx" in metrics
    assert metrics["xct"] >= 0.0