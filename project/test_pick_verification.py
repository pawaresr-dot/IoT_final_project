import unittest
from pick_verification import verify_follow,retry_allowed,choose_grasp_cycle

class PickChecks(unittest.TestCase):
    def test_disappearance_alone_is_unknown(self):
        self.assertEqual(verify_follow(False,[])['verdict'],'unknown')
    def test_source_object_overrides_spurious_arm_match(self):
        tracks=[{'score':.9,'center':p} for p in [(100,100),(130,100),(100,100)]]
        self.assertEqual(verify_follow(True,tracks)['verdict'],'not_picked')
    def test_returning_track_follows(self):
        tracks=[{'score':.9,'center':p} for p in [(100,100),(130,100),(102,100)]]
        self.assertEqual(verify_follow(False,tracks)['verdict'],'followed_visually')
    def test_stationary_or_weak_tracks_do_not_pass(self):
        for score,points,verdict in [(.9,[(100,100)]*3,'not_picked'),(.4,[(100,100),(130,100),(100,100)],'unknown')]:
            self.assertEqual(verify_follow(False,[{'score':score,'center':p} for p in points])['verdict'],verdict)
    def test_retry_is_bounded_and_requires_visible_aligned_failure(self):
        self.assertTrue(retry_allowed('not_picked',1,3,True))
        for verdict,attempt,aligned in [('unknown',1,True),('not_picked',3,True),('not_picked',1,False)]:
            self.assertFalse(retry_allowed(verdict,attempt,3,aligned))
    def test_last_complete_grasp_cycle_is_selected(self):
        goals=[2373,2300,2100,2200,2373,2360,2300,2100,2089,2089,2110,2250,2373]
        rows=[{'action_after_raw':[0]*5+[v]} for v in goals]
        grasp,closed,release,opened=choose_grasp_cycle(rows)
        self.assertEqual((grasp,closed,opened),(6,7,12))
        self.assertGreater(release,closed)

if __name__=='__main__':unittest.main()
