"""Experimental joint-space stepping, not collision or Cartesian planning."""
from bounded_trim import BoundedTrim


def make_transfer_route(poses, motors):
    outside = {m: int(poses['retreat']['raw_position'][m]) for m in motors}
    source = {m: int(poses['lift']['raw_position'][m]) for m in motors}
    aligned = dict(source)
    aligned['shoulder_pan'] = outside['shoulder_pan']
    return [('retreat', outside), ('outside_transfer_height', dict(aligned)),
            ('lift', source), ('outside_transfer_height', dict(aligned)),
            ('retreat', dict(outside))]


def make_source_route(poses, motors):
    transfer = make_transfer_route(poses, motors)
    descent = [(stage, {m: int(poses[stage]['raw_position'][m]) for m in motors})
               for stage in ('approach', 'grasp', 'approach', 'lift')]
    return transfer[:3] + descent + transfer[3:]


def make_return_route(poses, motors):
    transfer = make_transfer_route(poses, motors)
    return [(name, {m: int(poses[name]['raw_position'][m]) for m in motors})
            for name in ('grasp', 'approach', 'lift')] + transfer[3:]


class RouteStepper:
    def __init__(self, current, limits, max_step=2, independent=False):
        if type(max_step) is not int or not 1 <= max_step <= 4:
            raise ValueError('Unsupported bounded step size')
        self.nominal = dict(current)
        self.limits = limits
        self.max_step,self.independent=max_step,independent
        lo, hi = limits['elbow_flex']
        self.elbow = BoundedTrim(current['elbow_flex'], lo + 3, hi - 3)
        self.trimmers = {'elbow_flex': self.elbow}
        if 'shoulder_lift' in current:
            lo, hi = limits['shoulder_lift']
            # Observed loaded shoulder settles 7 ticks from nominal. Allow
            # 10 ticks (<0.9 degree), without increasing trim or command lead.
            self.trimmers['shoulder_lift'] = BoundedTrim(current['shoulder_lift'], lo + 3, hi - 3, tolerance=10)

    def step(self, target, actual):
        for m in self.nominal:
            lo, hi = self.limits[m]
            if not lo + 3 <= target[m] <= hi - 3:
                raise ValueError('Route target outside limits: ' + m)
            if not lo + 3 <= actual[m] <= hi - 3:
                raise ValueError('Measured position outside limits: ' + m)
        if self.independent:
            for m in self.nominal:
                # A streamed target may reverse or retreat before a loaded
                # joint catches up. Do not require reaching an obsolete goal
                # on the opposite side of the newly requested motion.
                if not min(actual[m],target[m]) <= self.nominal[m] <= max(actual[m],target[m]):
                    self.nominal[m]=actual[m]
                    if m in self.trimmers: self.trimmers[m].trim=0
        ready={m:abs(actual[m]-self.nominal[m]) <= (self.trimmers[m].tolerance if m in self.trimmers else 12)
               for m in self.nominal}
        for m in self.nominal:
            if (ready[m] if self.independent else all(ready.values())):
                self.nominal[m] += max(-self.max_step,min(self.max_step,target[m]-self.nominal[m]))
        command = dict(self.nominal)
        for m, trimmer in self.trimmers.items():
            trimmer.nominal = self.nominal[m]
            trimmer.saturate_at_limits = self.independent
            requested, _ = trimmer.step(actual[m])
            lo = max(trimmer.nominal - 32, actual[m] - 30, self.limits[m][0] + 3)
            hi = min(trimmer.nominal + 32, actual[m] + 30, self.limits[m][1] - 3)
            if lo > hi:
                raise ValueError('Route tracking stalled: ' + m)
            command[m] = max(lo, min(hi, requested))
            trimmer.trim = command[m] - trimmer.nominal
        for m, value in command.items():
            lo, hi = self.limits[m]
            if not lo + 3 <= value <= hi - 3:
                raise ValueError('Route command outside limits: ' + m)
            # Never let the compensation issue a large catch-up command.
            if abs(value - actual[m]) > 30:
                raise ValueError('Route tracking stalled: ' + m)
        reached = all(self.nominal[m] == target[m] and
                      abs(actual[m] - target[m]) <= (self.trimmers[m].tolerance if m in self.trimmers else 12)
                      for m in self.nominal)
        return command, reached
