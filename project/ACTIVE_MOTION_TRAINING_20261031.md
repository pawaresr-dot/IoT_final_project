# Active-motion retraining

The first autonomous trials moved the arm but stalled overhead without a jaw
close. The four original demonstrations begin with 18.7–37.3 seconds before
their first measured motion of at least four ticks. This is evidence of idle
data, not proof that idle data alone caused rollout failure.

Export now optionally retains rows within two seconds of a measured or issued
goal change of at least four ticks on any joint, including the gripper.
Removed idle intervals and image gaps over 0.5 seconds create episode
boundaries. No action chunk crosses those boundaries. Original captures and
earlier exported datasets are retained.

The active dataset contains 3,793 samples across 66 segments from the SAME
four operator-confirmed demonstrations, compared with 5,228 samples before.
Normalization roundtrip is exact and image/resampling timestamp distance
remains at most 0.25 seconds. This filtering does not create new independent
demonstrations or an independent evaluation set.

Config: `act_four_active_config.json`, fresh 1,800-step ACT model.
Dataset: `training_datasets/four_successes_active_20261031_320`.
Output: `training_outputs/act_four_active_20261031`.
Training and policy trials are separate. The trial runner uses select_action,
resets cached actions on camera gaps and jaw operations, and retains the
controller's calibrated limits and bounded motion. Added a stop when all
joint spans stay below six ticks for more than nine seconds after startup.

Training completed all 1,800 steps. Real execution at
`vision_state/ai_trial_20261031T190308` logged 409 samples, body motion,
and zero jaw-close commands. Object remained on the table. Stopped the
unsuccessful approach and retained controller holding/error=null. Idle
filtering alone did not yield a successful autonomous pickup.
Next useful data is operator correction from the policy's stalled pose.
