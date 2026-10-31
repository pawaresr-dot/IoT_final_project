# Visual following check and retry

The source correction recording includes two close/open attempts. The
earlier replay incorrectly treated the first close/open as the full pickup.
Cycle selection now uses the last complete closed interval: grasp514,
closed539, release780, fully open799. Operator success remains scoped to the
entire original recording; it does not independently certify each attempt.

After jaw close, the runner reverses the taught approach while keeping the
jaw closed, exposing the source for inspection. If the object remains near
its pre-grasp position, this is a failed pickup. It opens the jaw and retries
only if the object is still aligned with the taught fixed-point coverage.
Maximum total attempts is three. No random positional corrections are made.

Disappearance alone is unknown. A candidate successful visual following
check requires a strong object-template match (>=0.70) across an A-B-A
30-tick base probe: >=12 pixels of movement and <=10 pixels return error.
Weak/occluded/inconsistent matching stops in hold rather than triggering
another blind pickup or transfer. This is image-based following evidence,
not an independent height measurement. Templates can still be confused by
appearance, illumination, and occlusion; physical trials remain necessary.

Unit tests cover missing/weak evidence, stationary false matches, source
presence overriding a candidate, consistent following, bounded retry and
last-cycle selection. Six tests passed. Hardware operation of this new
retry sequence has not yet been verified.
