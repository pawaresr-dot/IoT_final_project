# Close/transfer/open ACT experiment

- Source: `training_captures/20261031T112940_697ea07c`
- Capture: 1168 original frames, 122.438 seconds, completed normally.
- Both closing and opening goal changes are recorded. Top-view contact-sheet
  review shows the object moving from the original cell to the right edge.
  The operator subsequently confirmed both lifting clear of the surface and
  release outside the grid. This is operator-confirmed success, not an
  independent image-based height measurement.
- Full-resolution normalized copy: `training_datasets/20261031T112940_normalized`
- Training copy: `training_datasets/20261031T112940_normalized_320`
- Training-copy images: 320x180 RGB, PIL bilinear. Raw images are preserved.
- Resampling: 1225 frames at 10 Hz, nearest original row; state and issued
  action stay paired with the selected image. Mapping is in capture_provenance.
- Units: calibrated degrees for five body joints, calibrated 0..100 for gripper.
  Calibration snapshot comes from the first normalized export, not a later
  independently changed calibration. Raw conversion roundtrip was exact.
- Fresh model initialization, not an overwrite of the earlier lift-only pilot.
- Config: `act_pick_config.json`; log: `act_pick_train.log`.
- Output: `training_outputs/act_pick_20261031_01`.
- Offline training on MPS; no Hub upload, no controller/serial writes.
- Dataset has only one episode. No independent validation/test result can be
  claimed. Training-episode prediction errors are diagnostic only.

## Live inference contract

## Completed training and audit (2026-10-31)

Fresh training finished at step 600, then resumed with optimizer/data-order
state to step 1800. Checkpoint `001800/pretrained_model` is saved alongside
earlier checkpoints; no model was deployed to the robot.

`vision_state/act_pick_audit_20261031_1800.json` records successful loading of
the saved model and processors, 24 training-episode predictions and 10 fresh
live-camera shadow predictions. No robot commands were sent. Training-sample
MAE by saved joint order: 3.295, 0.661, 0.584, 0.626, 0.062 degrees, then
0.692 gripper units. This is NOT held-out accuracy. Compared with step 600,
shoulder_lift/elbow/wrist/gripper improved but pan error did not improve
(3.086 to 3.295 degrees). Do not present this as uniformly improved accuracy
or autonomous pickup success. A separate full demonstration/evaluation episode
is needed; retain this run as a reproducible pilot.

Use PIL bilinear 320x180 RGB resizing, then the saved checkpoint processors.
Normalize observed raw servo values with the saved calibration snapshot and
the installed LeRobot `use_degrees=True` convention. The model's postprocessed
outputs are absolute calibrated goals, not encoder ticks or velocity commands.
Load state and action dimensions in saved feature order. Do not use leader
coordinates as follower state. Do not use the stale board homography to place
objects. This model is not yet connected to robot command execution.
