# Frozen-checkpoint evaluation: failure to generalize

Frozen model: `training_outputs/act_pick_20261031_01/checkpoints/001800/pretrained_model`.
Evaluation source: `training_captures/20261031T114929_d260fc60`.
Derivative: `training_datasets/20261031T114929_eval_320`.
Results: `vision_state/act_pick_heldout_20261031_01.json`.

The source stopped after 966 frames (101.343 seconds) when forwarding returned
HTTP409. The original incomplete flag/error remain intact. Explicit offline-only
export preserved paired image/state/issued-action samples and interruption
metadata, resampling to 1014 frames at10Hz. This source and derivative remain
excluded from training; no retraining or checkpoint selection used its samples.
Constant-mean comparison uses frozen TRAINING statistics, not evaluation means.

Twenty-four evenly spaced available samples were evaluated, not all frames and
not an autonomous rollout. Mean absolute prediction errors were:

| Joint | Error | Unit |
|---|---:|---|
| shoulder_pan |4.696|degrees|
| shoulder_lift |33.548|degrees|
| elbow_flex |21.110|degrees|
| wrist_flex |12.356|degrees|
| wrist_roll |0.527|degrees|
| gripper |9.981|calibrated 0..100|

Direct state-envelope comparison establishes substantial distribution shift:
training elbow observations span79.209..93.363deg, while this evaluation spans
41.231..65.758deg; all evaluation elbow observations are outside the training
range. Approximately95.2% of shoulder-lift observations are also outside its
training range. This supports inadequate pose coverage as a contributor, not
proof that it is the only cause. Normalization uses the same saved calibration.
Do not attempt to fix this failure by changing calibration to force overlap.

No robot commands were issued during evaluation. This model is not qualified
for autonomous pickup. Next: additional complete operator demonstrations covering
the new posture, followed by another untouched evaluation episode. Preserve this
evaluation as a failed frozen-model result; do not relabel it as a training success.
