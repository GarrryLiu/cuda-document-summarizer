import random
import unittest
from cuda_summary.scheduling import token_budget_batches,plan_stats,rounded,distinct_plans

class ScheduleTests(unittest.TestCase):
    def test_plan_aliases_preserve_order_and_batch_boundaries(self):
        plans={'a':[[0,1],[2]],'b':[[0,1],[2]],'c':[[2],[0,1]],'d':[[0],[1,2]]}
        unique,aliases=distinct_plans(plans)
        self.assertEqual(aliases,{'b':'a'})
        self.assertEqual(list(unique),['a','c','d'])
    def test_variable_capacity_and_tail(self):
        lengths=[500,100,300,100]
        p=token_budget_batches(lengths,1024,128,max_sequences=4)
        self.assertEqual(p,[[1,3],[2],[0]])
    def test_reserves_output_and_padding(self):
        p=token_budget_batches([257,257],800,256,max_sequences=8)
        self.assertEqual(p,[[0],[1]])
        self.assertEqual(plan_stats([257,257],p,256)['max_reserved_token_slots'],520)
    def test_oversize_fails_without_dropping(self):
        with self.assertRaises(ValueError): token_budget_batches([1000],1024,128)
    def test_random_coverage_and_bound(self):
        rng=random.Random(42)
        for n in range(1,80):
            lengths=[rng.randint(1,3000) for _ in range(n)]
            p=token_budget_batches(lengths,8192,512)
            self.assertEqual(sorted(i for b in p for i in b),list(range(n)))
            self.assertTrue(all(len(b)<=8 and len(b)*(rounded(max(lengths[i] for i in b))+512)<=8192 for b in p))
    def test_validation(self):
        for kwargs in [{'token_budget':0},{'output_reserve':-1},{'max_sequences':0}]:
            args={'token_budget':1000,'output_reserve':100};args.update(kwargs)
            with self.assertRaises(ValueError):token_budget_batches([100],**args)
        self.assertEqual(token_budget_batches([],1000,100),[])
