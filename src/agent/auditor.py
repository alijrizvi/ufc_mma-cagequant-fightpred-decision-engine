import re
from typing import Dict, Any, List, Tuple

class AnalyticalAuditor:
    """Receipt-style auditor verifying numerical claims against ground-truth data within tolerance."""

    def verify_claims(
        self, 
        text_response: str, 
        ground_truth: Dict[str, Any], 
        tolerance_pct: float = 1.5
    ) -> Tuple[bool, List[str]]:
        """Parses percentage claims and validates against expected metrics."""
        violations = []
        pct_matches = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", text_response)]
        
        val_a = ground_truth.get("win_rate_a", 0.0)
        val_b = ground_truth.get("win_rate_b", 0.0)
        
        expected_a = round(val_a * 100.0 if val_a <= 1.0 else val_a, 1)
        expected_b = round(val_b * 100.0 if val_b <= 1.0 else val_b, 1)
        valid_targets = [expected_a, expected_b]

        for val in pct_matches:
            matched = any(abs(val - target) <= tolerance_pct for target in valid_targets)
            if not matched and val not in [0.0, 100.0]:
                violations.append(f"Unverified numeric claim: {val}% does not match ground truth {valid_targets}.")

        is_valid = len(violations) == 0
        return is_valid, violations