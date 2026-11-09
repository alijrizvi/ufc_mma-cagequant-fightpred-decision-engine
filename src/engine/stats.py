from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class FighterCombatProfile:
    name: str
    # Attack Pillar
    punch_power: float = 20.0
    elbow_power: float = 25.0
    leg_kick_power: float = 15.0
    strike_hit_pct: float = 0.50
    
    # Defense Pillar
    arm_block_pct: float = 0.20
    leg_block_pct: float = 0.15
    takedown_def_pct: float = 0.75
    durability_index: float = 1.0  # Multiplier: >1.0 takes more damage, <1.0 chin toughness
    
    # Grappling Pillar
    takedown_success_pct: float = 0.40
    gnp_avg_damage: float = 12.0
    submission_conversion_pct: float = 0.15
    
    # Stamina Pillar
    stamina_decay_rate: float = 0.03  # % lost per high-output round

class CombatPillarEngine:
    """Calculates opponent-adjusted combat statistics across Attack, Defense, Grappling, and Stamina."""

    def calculate_interval_strike_damage(
        self, 
        attacker: FighterCombatProfile, 
        defender: FighterCombatProfile, 
        attacker_stamina: float = 100.0
    ) -> float:
        """Calculates net strike damage adjusted for defender mitigation and attacker stamina decay."""
        stamina_factor = max(0.1, min(1.0, attacker_stamina / 100.0))
        raw_power = attacker.punch_power * stamina_factor
        
        mitigation = max(0.0, min(0.95, defender.arm_block_pct))
        net_damage = raw_power * (1.0 - mitigation) * defender.durability_index
        
        return round(net_damage, 2)