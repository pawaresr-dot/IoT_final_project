# Single-episode ACT pipeline pilot

This pilot uses capture `20261031T110710_09b6c229`: 559 original images over
59.035 seconds, resampled with nearest-source-row selection to 591 samples at
10 Hz. The operator confirmed a successful lift. Release outside the grid was
not separately confirmed. No held-out evaluation episode is available.

## Reproduction

Run `lerobot-train --config_path=act_pilot_config.json` in the existing lerobot
environment. The configured output directory must not already contain a run;
choose a new output directory for a new experiment. Never overwrite checkpoints.

- Dataset: `training_datasets/20261031T110710_normalized`
- Configuration: `act_pilot_config.json`
- Log: `act_pilot_train.log`
- Outputs: `training_outputs/act_pilot_20261031_01`
- Offline training only; no robot port is opened and no Hub upload is enabled.
- ACT: small transformer, ResNet18 initialized without pretrained weights,
  16-action chunks, batch 2, 100 optimizer steps, MPS.
- Image preprocessing: RGB, resize to height 180 / width 320, then the saved
  policy processor normalization. Inference must use the same resize; training
  dataset transforms are NOT automatically a live-camera inference pipeline.
- State/action order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
  wrist_roll, gripper. Body values are calibrated degrees; gripper is calibrated
  0..100. Actions are issued follower goals, not leader poses or measured states.
- Calibration snapshot and SHA256 are stored with the normalized dataset.
- `accelerate==1.15.0` was added to the existing environment to enable training.

## Interpretation and restrictions

## Pilot result (2026-10-31)

Training exited successfully after 100 steps in approximately 68 seconds.
Checkpoint 000100 contains model.safetensors, saved pre/postprocessors and
optimizer state. Logged averaged training loss was 82.824 at step 10 and
30.050 at step 100 (different training batches, not held-out performance).
No autonomous robot rollout has been performed.

## Offline audit and continuation

`audit_act_policy.py` loaded checkpoint 000100 and its saved processors,
evaluated 24 evenly spaced samples from the TRAINING episode, and performed
10 live-camera shadow predictions. All predictions were finite; no commands
were sent. Results are in `vision_state/act_pilot_audit_20261031_01.json`.
At 100 steps, mean absolute body errors were approximately 4.03, 0.21, 2.14,
2.47 and 0.07 degrees; gripper error was 6.21 calibrated units. Holding the
observed position was a better baseline on several joints. This is not a
qualified autonomous controller.

Training was resumed from checkpoint 000100 to a total of 600 steps, preserving
optimizer/data-order state and saving checkpoints every 100 steps. Progress is
in `act_continued_train.log`. This continuation remains a lift/release pilot,
not complete pickup training or a live rollout.

Capture sequence audit found commanded gripper values starting at 2088 and
only increasing toward 2373 (opening). No closing motion is recorded in this
episode. Operator-confirmed successful lifting remains valid, but it does not
provide a complete approach/open/close/lift/release demonstration. The next
demonstration must begin BEFORE grasping. A read-only 3-second leader check
completed after reconnect; this is not proof that its intermittent USB fault
has been resolved.

This is a training-pipeline smoke test, not a validated autonomous policy.
Low training loss cannot establish grasp success or generalization. One episode
cannot support an independent test split. Do not report held-out accuracy.
Dataset provenance intentionally retains `training_ready=false` pending full
demonstration/timing review. Pilot use does not change that quality label.

Before live deployment, validate processor loading, offline finite predictions
and action units, implement bounded inference through the existing serial-owner
controller, and collect additional complete successful demonstrations with
varied object positions plus separate evaluation trials. Do not send policy
outputs directly to a second serial connection.
