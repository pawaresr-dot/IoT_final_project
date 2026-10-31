# Gripper representation mismatch

The earlier trial adapter waited for a predicted absolute goal below2200
before triggering a full jaw_close, and above2300 for jaw_open. The actual
training actions are short-lead issued servo goals from a gradual jaw motion.
There are zero training samples with measured gripper above2300 and issued
goal below2200. There are 278 samples with measured gripper above2300 and
issued goal at least five ticks below the measurement. Thus the old binary
adapter discards the initial incremental closing commands represented in
the data. This is a runner mismatch, not proof it caused all live failures.

At checkpoint6000, balanced samples of recorded closed/open states yield
32/32 matching absolute decisions per class. These are training states and
can reflect copying the already-closed/open observation; they do not prove
grasp initiation. The failed live rollout predicted raw2384..2389 while the
actual open gripper was2371, so those predictions themselves do not request
closing under either adapter.

Changed the runner to use bounded predicted-minus-measured grip direction:
16-tick deadband, three agreeing observations, at most11 ticks per jog.
The existing controller validates and executes each jog. Body streaming
pauses while the gripper jog runs. Cached ACT actions reset during jaw
motion. Existing calibrated bounds remain enforced.

Actual trial after this change logged164 samples and stopped on policy
stall; controller remained holding with no error/calibration_error.
Pickup is still unsuccessful. Fixing the representation mismatch did not
resolve the policy's failure to approach the object.
