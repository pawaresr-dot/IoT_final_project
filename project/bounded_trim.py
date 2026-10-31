"""Bounded experimental outer-loop position trim. No hardware access."""
from dataclasses import dataclass


@dataclass
class BoundedTrim:
    nominal: int
    lower: int
    upper: int
    trim: int = 0
    max_trim: int = 32
    tolerance: int = 6
    saturate_at_limits: bool = False

    def step(self, actual: int) -> tuple[int, bool]:
        if not self.lower <= actual <= self.upper:
            raise ValueError('Measured position outside joint limits')
        error = self.nominal - actual
        if abs(error) > self.tolerance:
            proposed = self.trim + (1 if error > 0 else -1)
            proposed = max(-self.max_trim, min(self.max_trim, proposed))
            command = self.nominal + proposed
            if not self.lower <= command <= self.upper:
                if not self.saturate_at_limits:
                    raise ValueError('Trim command would cross a joint limit')
                proposed = max(self.lower, min(self.upper, command)) - self.nominal
            self.trim = proposed
        return self.nominal + self.trim, abs(error) <= self.tolerance
