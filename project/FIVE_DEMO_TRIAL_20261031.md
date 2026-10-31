# Five-demonstration correction experiment

Combined the four active-motion datasets and the operator-confirmed
correction capture `20261031T190513_a9654b10`. Dataset contains 4,929 unique
samples in 116 segments from five demonstrations. Calibrations matched.
Fresh ACT training completed 1,800 steps using `act_five_active_config.json`.
Checkpoint: `training_outputs/act_five_active_20261031/checkpoints/001800/pretrained_model`.

Operator placed the keychain face up in the lower-left cell. Moved the arm
through the monitored controller to the first observed body pose of the
correction demonstration. Alignment completed within 15 ticks per joint.
The leader forwarding process was stopped before policy execution.

Trial: `vision_state/ai_trial_20261031T191334`.
The policy issued body movement, then stalled overhead. No jaw-close command
occurred; the object remained on the table. Trial stopped automatically at
277 samples after all joint spans stayed under six ticks for over nine
seconds. Autonomous pickup failed. This is an observed failure, not a
servo/connection fault or a successful grasp.

The latest correction did not resolve the policy's fixed-point behavior.
Do not promise that another arbitrary small number of demonstrations or
extra training steps will produce successful autonomy. Next investigation
should measure visual dependence, future action errors, and performance
from demonstration start states, or use a separately labelled vision plus
verified trajectory baseline for the submission.
