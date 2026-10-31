# Continued training to step 6000

Resumed the five-active model from checkpoint 001800, retaining optimizer
and data-order state. Completed step 6000 successfully, with snapshots at
2400, 3600, 4800, and 6000. Log: `act_five_active_continued.log`.
Training took approximately 10 minutes 40 seconds for 4200 additional steps.

Repeated the same 32-observation offline intervention/future-action audit
at all snapshots. This is a training-data diagnostic, not held-out accuracy.
On the 21 moving observations, horizon15 errors were:

| Step | Pan ° | Lift ° | Elbow ° | Wrist ° | Roll ° | Grip units |
| --- | --- | --- | --- | --- | --- | --- |
| 1800 | 3.408 | 5.225 | 2.878 | 2.354 | 0.214 | 1.632 |
| 2400 | 3.597 | 4.190 | 2.993 | 2.358 | 0.177 | 1.724 |
| 3600 | 2.533 | 4.169 | 2.674 | 1.733 | 0.226 | 1.327 |
| 4800 | 2.769 | 3.018 | 2.468 | 1.455 | 0.152 | 1.067 |
| 6000 | 2.524 | 3.048 | 2.286 | 1.555 | 0.130 | 1.421 |

The final snapshot improves all columns compared with 1800, though not
monotonically and not uniformly better than holding on all axes.

Real execution of checkpoint006000 logged 151 samples and stopped on the
policy-stall condition around 20 seconds. The keychain remained on the
table and no jaw-close command occurred. The controller retained holding
with error=null. Autonomous pickup failed. Longer optimization improved
this offline metric but did not fix the observed live policy fixed point.
Do not treat further arbitrary training duration as a demonstrated fix.
