import numpy as np
DATA_PROVENANCE = "synthetic"  # shown in the UI: outcomes are simulated, not real bouts
import duckdb
import pandas as pd
from typing import Dict, Any, Optional
from src.engine.stats import FighterCombatProfile

class OctagonDataEngine:
    """Embedded DuckDB analytics and feature storage engine for UFC Lightweight fight telemetry."""

    def __init__(self, db_path: str = ":memory:"):
        self.conn = duckdb.connect(db_path)
        self._initialize_tables()

    def _initialize_tables(self):
        """Initializes relational tables for fighter baselines and 10-second interval telemetry."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS fighter_profiles (
                fighter_name VARCHAR PRIMARY KEY,
                punch_power DOUBLE,
                elbow_power DOUBLE,
                leg_kick_power DOUBLE,
                strike_hit_pct DOUBLE,
                arm_block_pct DOUBLE,
                leg_block_pct DOUBLE,
                takedown_def_pct DOUBLE,
                durability_index DOUBLE,
                takedown_success_pct DOUBLE,
                gnp_avg_damage DOUBLE,
                submission_conversion_pct DOUBLE,
                stamina_decay_rate DOUBLE
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS interval_telemetry (
                interval_id INTEGER,
                fighter_name VARCHAR,
                round_num INTEGER,
                strikes_attempted INTEGER,
                strikes_landed INTEGER,
                damage_dealt DOUBLE,
                clinch_seconds INTEGER,
                ground_seconds INTEGER,
                opponent_td_attempted INTEGER,
                opponent_td_landed INTEGER,
                grapple_exchanges INTEGER,
                grapple_positions_won INTEGER
            )
        """)

    def seed_simulated_telemetry(self):
        """Backward-compatible seeding method for test_metrics.py."""
        self.seed_historical_lightweight_dataset()

    def seed_historical_lightweight_dataset(self):
        """Populates historical baseline stats for UFC Lightweight division competitors."""
        self.conn.execute("DELETE FROM fighter_profiles")
        self.conn.execute("DELETE FROM interval_telemetry")

        profiles_data = [
            ('Islam Makhachev', 24.0, 22.0, 14.0, 0.60, 0.48, 0.40, 0.90, 0.85, 0.62, 14.0, 0.25, 0.025),
            ('Carlos Prates', 26.0, 28.0, 18.0, 0.54, 0.35, 0.30, 0.55, 1.10, 0.42, 16.0, 0.35, 0.045),
            ('Dustin Poirier', 28.0, 20.0, 16.0, 0.51, 0.42, 0.35, 0.64, 0.95, 0.36, 12.0, 0.12, 0.035),
            ('Justin Gaethje', 30.0, 18.0, 22.0, 0.49, 0.38, 0.25, 0.75, 0.90, 0.25, 10.0, 0.05, 0.040)
        ]
        
        self.conn.executemany("""
            INSERT INTO fighter_profiles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, profiles_data)

        telemetry_data = [
            (1, 'Islam Makhachev', 1, 8, 5, 14.0, 4, 6, 1, 0, 3, 3),
            (2, 'Islam Makhachev', 1, 10, 7, 18.0, 2, 8, 0, 0, 4, 3),
            (1, 'Carlos Prates', 1, 12, 6, 16.0, 1, 0, 2, 1, 3, 1),
            (2, 'Carlos Prates', 1, 14, 7, 20.0, 0, 0, 1, 0, 4, 1)
        ]

        self.conn.executemany("""
            INSERT INTO interval_telemetry VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, telemetry_data)

    def load_fighter_profile(self, fighter_name: str) -> FighterCombatProfile:
        """Queries DuckDB for a fighter's profile and returns a typed FighterCombatProfile object."""
        row = self.conn.execute("""
            SELECT fighter_name, punch_power, elbow_power, leg_kick_power, strike_hit_pct,
                   arm_block_pct, leg_block_pct, takedown_def_pct, durability_index,
                   takedown_success_pct, gnp_avg_damage, submission_conversion_pct, stamina_decay_rate
            FROM fighter_profiles
            WHERE fighter_name = ?
        """, [fighter_name]).fetchone()

        if not row:
            raise ValueError(f"Fighter '{fighter_name}' not found in DuckDB database.")

        return FighterCombatProfile(
            name=row[0],
            punch_power=row[1],
            elbow_power=row[2],
            leg_kick_power=row[3],
            strike_hit_pct=row[4],
            arm_block_pct=row[5],
            leg_block_pct=row[6],
            takedown_def_pct=row[7],
            durability_index=row[8],
            takedown_success_pct=row[9],
            gnp_avg_damage=row[10],
            submission_conversion_pct=row[11],
            stamina_decay_rate=row[12]
        )

    def get_advanced_feature_metrics(self, fighter_name: str) -> Dict[str, float]:
        """Calculates advanced analytical metrics (xCT, Stamina Drain Ratio, Strike Volatility Index)."""
        query = """
            SELECT 
                AVG(clinch_seconds + ground_seconds) / 10.0 AS xct,
                AVG(damage_dealt) * 0.05 AS stamina_drain_ratio,
                STDDEV_SAMP(damage_dealt) AS strike_volatility_idx
            FROM interval_telemetry
            WHERE fighter_name = ?
        """
        result = self.conn.execute(query, [fighter_name]).fetchone()

        return {
            "xct": round(result[0] or 0.0, 4),
            "stamina_drain_ratio": round(result[1] or 0.0, 4),
            "strike_volatility_idx": round(result[2] or 0.0, 4)
        }

    def get_fighter_pillar_metrics(self, fighter_name: str) -> Dict[str, float]:
        """Executes governed semantic queries over DuckDB telemetry."""
        query = """
            SELECT 
                SUM(strikes_landed) * 1.0 / NULLIF(SUM(strikes_attempted), 0) AS strike_hit_pct,
                SUM(damage_dealt) * 1.0 / NULLIF(SUM(strikes_landed), 0) AS avg_damage_per_strike,
                1.0 - (SUM(opponent_td_landed) * 1.0 / NULLIF(SUM(opponent_td_attempted), 0)) AS takedown_def_pct,
                SUM(grapple_positions_won) * 1.0 / NULLIF(SUM(grapple_exchanges), 0) AS grapple_win_pct
            FROM interval_telemetry
            WHERE fighter_name = ?
        """
        result = self.conn.execute(query, [fighter_name]).fetchone()

        return {
            "strike_hit_pct": round(result[0] or 0.50, 4) if result and result[0] is not None else 0.50,
            "avg_damage_per_strike": round(result[1] or 10.0, 2) if result and result[1] is not None else 10.0,
            "takedown_def_pct": round(result[2] or 0.75, 4) if result and result[2] is not None else 0.75,
            "grapple_win_pct": round(result[3] or 0.50, 4) if result and result[3] is not None else 0.50
        }

    def get_training_frame(
        self, n_fights: int = 1000, intervals_per_fight: int = 6, seed: int = 42
    ) -> pd.DataFrame:
        """Builds a SYNTHETIC grouped training frame from fighter profiles."""
        rng = np.random.default_rng(seed)
        rows = []
        for fight_id in range(1, n_fights + 1):
            advantage = rng.normal(0.0, 1.0)
            logit = 1.4 * advantage + rng.normal(0.0, 0.5)
            p_win = 1.0 / (1.0 + np.exp(-logit))
            winner = int(rng.random() < p_win)
            hp_a, hp_b = 100.0, 100.0
            stam_a, stam_b = 100.0, 100.0
            for step in range(1, intervals_per_fight + 1):
                dmg_a = max(0.0, rng.normal(8.0 + 2.5 * advantage, 3.0))
                dmg_b = max(0.0, rng.normal(8.0 - 2.5 * advantage, 3.0))
                hp_a = max(0.0, hp_a - dmg_b)
                hp_b = max(0.0, hp_b - dmg_a)
                stam_a = max(10.0, stam_a - rng.uniform(1.5, 3.5))
                stam_b = max(10.0, stam_b - rng.uniform(1.5, 3.5))
                rows.append(
                    {
                        "fight_id": fight_id,
                        "round_num": (step - 1) // 2 + 1,
                        "fighter_a_hp": hp_a,
                        "fighter_b_hp": hp_b,
                        "fighter_a_stamina": stam_a,
                        "fighter_b_stamina": stam_b,
                        "damage_diff": dmg_a - dmg_b,
                        "power_diff": advantage * 5.0,
                        "hp_delta": hp_a - hp_b,
                        "stamina_delta": stam_a - stam_b,
                        "bout_winner": winner,
                    }
                )
        return pd.DataFrame(rows)
