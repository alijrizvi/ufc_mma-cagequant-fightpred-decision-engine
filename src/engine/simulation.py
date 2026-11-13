import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.engine.etl import OctagonDataEngine
from src.engine.models import FightPredictorXGB
from src.engine.stats import CombatPillarEngine, FighterCombatProfile
from src.engine.train import train_xgboost_predictor


@dataclass
class FighterState:
    name: str
    profile: FighterCombatProfile
    base_hp: float = 100.0
    hp: float = 100.0
    stamina: float = 100.0

    def __post_init__(self):
        if self.hp == 100.0 and self.base_hp != 100.0:
            self.hp = self.base_hp

@dataclass
class IntervalLog:
    interval_id: int
    round_num: int
    time_seconds: int
    fighter_a_hp: float
    fighter_b_hp: float
    action_type: str
    damage_dealt_a: float
    damage_dealt_b: float
    p_win_a: float = 0.50
    p_win_b: float = 0.50

@dataclass
class SimulationResult:
    winner: str
    method_of_victory: str
    finish_round: int
    finish_time_str: str
    total_intervals: int
    scores: Dict[str, List[int]]
    interval_logs: List[IntervalLog]

_MODEL_CACHE: dict[int, tuple] = {}


def _cached_predictor(db_engine):
    """Trains the XGBoost predictor once per distinct training frame, then reuses it.

    Without this, every Monte Carlo run (and every Streamlit rerun) retrained the model.
    """
    frame = db_engine.get_training_frame()
    key = int(pd.util.hash_pandas_object(frame, index=False).sum())
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = train_xgboost_predictor(frame)
    return _MODEL_CACHE[key]


class OctagonSimulationEngine:
    """Core 10-second discrete interval simulation engine with stochastic variance and XGBoost integration."""

    def __init__(
        self, 
        db_engine: Optional[OctagonDataEngine] = None, 
        predictor: Optional[FightPredictorXGB] = None
    ):
        self.pillar_engine = CombatPillarEngine()
        
        if predictor is not None:
            self.predictor = predictor
            self.brier_score = 0.0
        else:
            if db_engine is None:
                db_engine = OctagonDataEngine()
                db_engine.seed_historical_lightweight_dataset()
            self.predictor, self.brier_score = _cached_predictor(db_engine)

    def apply_interval_damage(
        self, 
        fighter: FighterState, 
        raw_damage: float, 
        mitigation: float = 0.0
    ) -> FighterState:
        """Applies damage to fighter HP with defensive mitigation, enforcing a non-negative floor (HP >= 0.0)."""
        effective_mitigation = max(0.0, min(1.0, mitigation))
        net_damage = raw_damage * (1.0 - effective_mitigation)
        fighter.hp = max(0.0, fighter.hp - net_damage)
        return fighter

    def score_round(self, round_damage_a: float, round_damage_b: float) -> Tuple[int, int]:
        """Calculates official 10-9 (or 10-8) round scoring based on net damage differential."""
        diff = round_damage_a - round_damage_b
        if abs(diff) < 2.0:
            return (10, 10)
        elif diff >= 25.0:
            return (10, 8)
        elif diff > 0:
            return (10, 9)
        elif diff <= -25.0:
            return (8, 10)
        else:
            return (9, 10)

    def run_full_fight(
        self, 
        profile_a: FighterCombatProfile, 
        profile_b: FighterCombatProfile, 
        max_rounds: int = 5,
        stochastic: bool = True
    ) -> SimulationResult:
        """Simulates bout progression with dynamic P_win estimation per 10-second interval."""
        fighter_a = FighterState(name=profile_a.name, profile=profile_a)
        fighter_b = FighterState(name=profile_b.name, profile=profile_b)
        
        interval_logs: List[IntervalLog] = []
        round_damage_accumulator: Dict[int, Dict[str, float]] = {}
        total_intervals_max = max_rounds * 30

        method_of_victory = "Decision (Unanimous)"
        finish_round = max_rounds
        finish_time_str = "5:00"

        for interval_id in range(1, total_intervals_max + 1):
            round_num = ((interval_id - 1) // 30) + 1
            sec_in_round = ((interval_id - 1) % 30) * 10

            if round_num not in round_damage_accumulator:
                round_damage_accumulator[round_num] = {"a": 0.0, "b": 0.0}

            # Calculate base strike damage
            raw_dmg_a = self.pillar_engine.calculate_interval_strike_damage(
                attacker=fighter_a.profile, defender=fighter_b.profile, attacker_stamina=fighter_a.stamina
            )
            raw_dmg_b = self.pillar_engine.calculate_interval_strike_damage(
                attacker=fighter_b.profile, defender=fighter_a.profile, attacker_stamina=fighter_b.stamina
            )

            # Apply stochastic Gaussian volatility (modeling clean connections / slip variance)
            if stochastic:
                mult_a = max(0.1, random.gauss(1.0, 0.35))
                mult_b = max(0.1, random.gauss(1.0, 0.35))
                damage_a_to_b = raw_dmg_a * mult_a
                damage_b_to_a = raw_dmg_b * mult_b
            else:
                damage_a_to_b = raw_dmg_a
                damage_b_to_a = raw_dmg_b

            self.apply_interval_damage(fighter_b, raw_damage=damage_a_to_b, mitigation=0.0)
            self.apply_interval_damage(fighter_a, raw_damage=damage_b_to_a, mitigation=0.0)

            round_damage_accumulator[round_num]["a"] += damage_a_to_b
            round_damage_accumulator[round_num]["b"] += damage_b_to_a

            # Stamina decay
            fighter_a.stamina = max(10.0, fighter_a.stamina - (profile_a.stamina_decay_rate * 0.5))
            fighter_b.stamina = max(10.0, fighter_b.stamina - (profile_b.stamina_decay_rate * 0.5))

            action_type = "Striking Exchange"

            # XGBoost win probability state vector
            interval_state = {
                "round_num": round_num,
                "time_seconds": sec_in_round,
                "fighter_a_hp": fighter_a.hp,
                "fighter_b_hp": fighter_b.hp,
                "fighter_a_stamina": fighter_a.stamina,
                "fighter_b_stamina": fighter_b.stamina,
                "damage_dealt_a": damage_a_to_b,
                "damage_dealt_b": damage_b_to_a
            }
            p_win_a, p_win_b = self.predictor.predict_interval_win_probability(
                interval_state, profile_a, profile_b
            )

            # Check KO / TKO Condition
            if fighter_b.hp <= 0.0 or fighter_a.hp <= 0.0:
                method_of_victory = "KO / TKO (Striking)"
                finish_round = round_num
                finish_time_str = f"{sec_in_round // 60}:{sec_in_round % 60:02d}"
                interval_logs.append(
                    IntervalLog(
                        interval_id, round_num, sec_in_round, 
                        fighter_a.hp, fighter_b.hp, "KO Finish", 
                        damage_a_to_b, damage_b_to_a, 
                        1.0 if fighter_b.hp <= 0.0 else 0.0,
                        0.0 if fighter_b.hp <= 0.0 else 1.0
                    )
                )
                break

            # Check Submission Condition
            if interval_id % 3 == 0:
                action_type = "Grappling Exchange"
                if fighter_b.hp < 40.0 or fighter_b.stamina < 30.0:
                    if random.random() < profile_a.submission_conversion_pct:
                        fighter_b.hp = 0.0
                        method_of_victory = "Submission (Tapout)"
                        finish_round = round_num
                        finish_time_str = f"{sec_in_round // 60}:{sec_in_round % 60:02d}"
                        interval_logs.append(
                            IntervalLog(
                                interval_id, round_num, sec_in_round, 
                                fighter_a.hp, 0.0, "Submission Lock", 
                                damage_a_to_b, damage_b_to_a, 1.0, 0.0
                            )
                        )
                        break

            interval_logs.append(
                IntervalLog(
                    interval_id, round_num, sec_in_round, 
                    fighter_a.hp, fighter_b.hp, action_type, 
                    damage_a_to_b, damage_b_to_a, p_win_a, p_win_b
                )
            )

        # Scoring Logic
        if fighter_a.hp <= 0.0:
            winner = fighter_b.name
        elif fighter_b.hp <= 0.0:
            winner = fighter_a.name
        else:
            scores_a, scores_b = [], []
            for r in range(1, finish_round + 1):
                dmg_a = round_damage_accumulator[r]["a"]
                dmg_b = round_damage_accumulator[r]["b"]
                sa, sb = self.score_round(dmg_a, dmg_b)
                scores_a.append(sa)
                scores_b.append(sb)

            total_a, total_b = sum(scores_a), sum(scores_b)
            if total_a > total_b:
                winner = fighter_a.name
                method_of_victory = f"Unanimous Decision ({total_a}-{total_b})"
            elif total_b > total_a:
                winner = fighter_b.name
                method_of_victory = f"Unanimous Decision ({total_b}-{total_a})"
            else:
                winner = "Draw"
                method_of_victory = f"Split Draw ({total_a}-{total_b})"

        scores_dict = {
            profile_a.name: [round_damage_accumulator[r]["a"] for r in round_damage_accumulator],
            profile_b.name: [round_damage_accumulator[r]["b"] for r in round_damage_accumulator]
        }

        return SimulationResult(
            winner=winner,
            method_of_victory=method_of_victory,
            finish_round=finish_round,
            finish_time_str=finish_time_str,
            total_intervals=len(interval_logs),
            scores=scores_dict,
            interval_logs=interval_logs
        )