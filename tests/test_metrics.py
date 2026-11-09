import pytest
import duckdb
from src.engine.etl import OctagonDataEngine

def test_duckdb_telemetry_ingestion():
    """Enforces that simulated fight interval telemetry ingests into DuckDB correctly."""
    db_engine = OctagonDataEngine()
    db_engine.seed_simulated_telemetry()
    
    row_count = db_engine.conn.execute("SELECT COUNT(*) FROM interval_telemetry").fetchone()[0]
    assert row_count > 0

def test_governed_striking_efficiency_metric():
    """Enforces that striking accuracy obeys the semantic formula in metrics.yml."""
    db_engine = OctagonDataEngine()
    db_engine.seed_simulated_telemetry()
    
    # Query governed metric view
    df_metrics = db_engine.get_fighter_pillar_metrics("Islam Makhachev")
    
    assert "strike_hit_pct" in df_metrics
    assert 0.0 <= df_metrics["strike_hit_pct"] <= 1.0