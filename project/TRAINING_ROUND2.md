# Second successful demonstration, 2026-10-31

Source: `training_captures/20261031T180447_81f2160b`.
Operator confirmed lift and release outside the grid with “สำเร็จด้วยดี”.
The recording contains 1,398 JPEGs over 193.357 seconds, including closing
and opening issued gripper goals. All JPEGs passed decoding checks.

There are 26 image gaps above 0.5 seconds (maximum 1.190 seconds).
Export splits at these gaps and resamples within each segment at 10 Hz.
This produces 1,766 paired image/state/issued-action samples across 27
segments. These are fragments of ONE demonstration, not 27 independent
successful trials. ACT action chunks stop at episode boundaries.
Missing motion is not reconstructed across these gaps. The largest selected
image-to-resampling timestamp distance was verified below 0.25 seconds.

Dataset: `training_datasets/20261031T180447_segmented_320`.
Images are 320x180 using PIL bilinear; original recordings are retained.
The calibration snapshot from the earlier training export is reused.
All original states and issued goals pass exact raw/normalized/raw roundtrip.

Config: `act_pick_round2_config.json`.
Output: `training_outputs/act_pick_20261031_round2`.
Log: `act_pick_round2_train.log`.
This is a fresh 600-step diagnostic model using only this second recording;
it does not combine the two successful demonstrations or overwrite the
earlier model. Training-sample audit is not independent validation.
No model command execution is enabled by this training workflow.
