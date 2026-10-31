# Controlled input diagnostic

Checkpoint: five-active demonstration model, step 1800.
Results: `vision_state/act_vision_diagnosis_20261031.json`.
32 training-data observations with 16 future actions in the same segment.
Repeated identical inference has zero difference, so interventions are
deterministic. No robot commands were sent.

With state fixed, swapping another real training image changes average
body targets by 8.45, 8.65, 10.06, 11.31, 0.97 degrees. Replacing the image
with gray changes them by 3.93, 5.39, 6.08, 6.82, 0.45 degrees.
Thus the policy uses visual input. This does NOT establish that it detects
the keychain or understands its location: swapped images also change arm
appearance, shadows, viewpoint, and background.

On the 21 sampled observations with meaningful upcoming motion, horizon15
policy error is 3.41, 5.23, 2.88, 2.35, 0.21 degrees and 1.63 gripper units.
Holding the observed position gives 2.57, 2.21, 2.42, 0.77, 0.31 degrees
and 1.95 gripper units. Policy is worse on four main body axes even on
training examples. This is evidence of inadequate learned action prediction,
not validation success and not a camera/USB/torque fault.

Training exposure: 1800 optimizer steps * batch2 = 3600 sampled frames,
against 4929 dataset frames, about 0.73 dataset equivalents. This is not
a claim that every frame was seen once; sampling and episode filtering
affect actual coverage. A fair next experiment is longer optimization on
the existing data, with checkpoint comparisons on the same diagnostic and
separate live trials, before requesting more arbitrary demonstrations.
No evidence here proves that more training alone will solve pickup.
