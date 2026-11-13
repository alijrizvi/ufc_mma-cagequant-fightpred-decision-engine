from dataclasses import dataclass
from enum import Enum
from typing import Dict

from src.engine.stats import FighterCombatProfile

# Like Dragon Ball Z (probably my most Nostalgic and Fav Anime - Tied now with ATLA and ofc Pokemon)
# I wanted ot introduce simulated "Combat Stance States" akin to Super Saiyan forms
# Understanding power progressions and form changes and the benefits/drawbacks of each Super Saiyan form requires great Fight IQ

class TacticalStance(Enum):
    STANDARD = "Standard Baseline"
    PRESSURE_BLITZ = "Power Overdrive (Grade 3)"
    COUNTER_FLOW = "Mastered Flow (Grade 4)"
    ASCENDED_SSJ2 = "Ascended SSJ2"

@dataclass
class StanceMultipliers:
    power_multiplier: float
    stamina_decay_multiplier: float
    block_multiplier: float
    accuracy_multiplier: float

class TacticalStanceEngine:
    """Manages tactical combat stance shifts, power scaling tradeoffs, and parameter adjustments."""

    STANCE_MAP: Dict[TacticalStance, StanceMultipliers] = {
        TacticalStance.STANDARD: StanceMultipliers(
            power_multiplier=1.00,
            stamina_decay_multiplier=1.00,
            block_multiplier=1.00,
            accuracy_multiplier=1.00
        ),
        TacticalStance.PRESSURE_BLITZ: StanceMultipliers(
            power_multiplier=1.40,
            stamina_decay_multiplier=2.50,
            block_multiplier=0.75,
            accuracy_multiplier=0.85
        ),
        TacticalStance.COUNTER_FLOW: StanceMultipliers(
            power_multiplier=1.05,
            stamina_decay_multiplier=0.60,
            block_multiplier=1.00,
            accuracy_multiplier=1.05
        ),
        TacticalStance.ASCENDED_SSJ2: StanceMultipliers(
            power_multiplier=1.35,
            stamina_decay_multiplier=1.10,
            block_multiplier=1.10,
            accuracy_multiplier=1.10
        )
    }

    def apply_stance_modifiers(
        self, 
        profile: FighterCombatProfile, 
        stance: TacticalStance
    ) -> FighterCombatProfile:
        """Returns a modified copy of FighterCombatProfile scaled by the active tactical stance."""
        mults = self.STANCE_MAP[stance]

        return FighterCombatProfile(
            name=profile.name,
            punch_power=profile.punch_power * mults.power_multiplier,
            elbow_power=profile.elbow_power * mults.power_multiplier,
            leg_kick_power=profile.leg_kick_power * mults.power_multiplier,
            strike_hit_pct=min(0.95, profile.strike_hit_pct * mults.accuracy_multiplier),
            arm_block_pct=min(0.90, profile.arm_block_pct * mults.block_multiplier),
            leg_block_pct=min(0.90, profile.leg_block_pct * mults.block_multiplier),
            takedown_def_pct=profile.takedown_def_pct,
            durability_index=profile.durability_index,
            takedown_success_pct=profile.takedown_success_pct,
            gnp_avg_damage=profile.gnp_avg_damage,
            submission_conversion_pct=profile.submission_conversion_pct,
            stamina_decay_rate=profile.stamina_decay_rate * mults.stamina_decay_multiplier
        )

    def determine_optimal_stance(
        self, 
        hp: float, 
        stamina: float, 
        preferred_mode: str = "balanced"
    ) -> TacticalStance:
        """Determines the optimal tactical stance given current vitality HP and stamina reserves."""
        if hp > 60.0 and stamina > 70.0:
            return TacticalStance.ASCENDED_SSJ2
        elif stamina < 35.0:
            return TacticalStance.COUNTER_FLOW
        elif preferred_mode == "power" and stamina >= 50.0:
            return TacticalStance.PRESSURE_BLITZ
        elif preferred_mode == "flow":
            return TacticalStance.COUNTER_FLOW
        else:
            return TacticalStance.STANDARD