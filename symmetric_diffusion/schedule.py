"""
Strength schedule parser for symmetrization.

Format:  "<start>-><end>@<hold>"

Examples:
    "1.0"              constant 1.0 throughout
    "1.0->0.0"         linear decay from 1.0 to 0.0 over all steps
    "1.0->0.2@0.7"     hold at 1.0 for first 70% of steps, then linear decay to 0.2
    "0.8->0.3@0.5"     hold at 0.8 for first 50%, then decay to 0.3
    "0.5->0.5"         same as "0.5" (constant)

The 'progress' parameter t goes from 0.0 (first step) to 1.0 (last step).
"""

from dataclasses import dataclass


@dataclass
class StrengthSchedule:
    """Piecewise-linear strength schedule."""
    start: float      # strength during hold phase
    end: float        # strength at the final step
    hold_until: float  # fraction of steps to hold at 'start' before decaying

    def __call__(self, t: float) -> float:
        """Evaluate strength at progress t in [0, 1].

        Args:
            t: Progress through the denoising process (0 = first step, 1 = last step).
        """
        if t <= self.hold_until:
            return self.start
        if self.hold_until >= 1.0:
            return self.start
        # Linear interpolation in the decay phase
        decay_progress = (t - self.hold_until) / (1.0 - self.hold_until)
        return self.start + (self.end - self.start) * decay_progress

    def __repr__(self):
        if self.start == self.end:
            return f"StrengthSchedule({self.start})"
        return f"StrengthSchedule({self.start}->{self.end}@{self.hold_until})"


def parse_schedule(s: str) -> StrengthSchedule:
    """Parse a schedule string.

    Formats:
        "1.0"              → constant
        "1.0->0.2"         → linear decay over all steps
        "1.0->0.2@0.7"     → hold then decay
    """
    s = s.strip()

    if "->" not in s:
        val = float(s)
        return StrengthSchedule(start=val, end=val, hold_until=1.0)

    parts = s.split("->")
    if len(parts) != 2:
        raise ValueError(f"Invalid schedule format: '{s}'. Use 'start->end' or 'start->end@hold'")

    start = float(parts[0])
    rest = parts[1]

    if "@" in rest:
        end_str, hold_str = rest.split("@")
        end = float(end_str)
        hold = float(hold_str)
    else:
        end = float(rest)
        hold = 0.0  # decay starts immediately

    if not (0 <= hold <= 1):
        raise ValueError(f"Hold fraction must be in [0, 1], got {hold}")

    return StrengthSchedule(start=start, end=end, hold_until=hold)
