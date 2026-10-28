import unittest
from cuda_summary.legal_quality import durations,triage,sensitive_changes


class QualityTriageTests(unittest.TestCase):
    def test_sensitive_negation_change_is_queued(self):
        self.assertEqual(len(sensitive_changes('May not disclose.','May disclose.')),1)
        self.assertEqual(sensitive_changes('May not disclose.','May not disclose.'),[])
        self.assertEqual(sensitive_changes('Heading A','Heading B'),[])
    def test_duration_spellings_and_hyphen(self):
        self.assertEqual(durations('three (3) years, 30-day notice; two months'),
                         {(3,'','year'),(30,'','day'),(2,'','month')})

    def test_qualifiers_and_changed_duration_are_flags(self):
        r=triage('five business days and three years','5 calendar days and 4 years')
        self.assertEqual(r['duration_mismatch_count'],2)
        self.assertTrue(r['review_required'])

    def test_contract_parentheses_and_line_breaks(self):
        self.assertFalse(triage('eighteen\n(18) months and thirty (30) days',
                                '18 months; 30 days')['review_required'])
        self.assertEqual(durations('twenty-one business days'),{(21,'business','day')})

    def test_no_flag_does_not_mean_entailment(self):
        # Same values with swapped subjects cannot be validated lexically.
        r=triage('Return in 3 days. Secrecy for 5 years.',
                 'Return in 5 years. Secrecy for 3 days.')
        self.assertFalse(r['review_required'])

    def test_numeric_anchor_and_word_budget(self):
        self.assertEqual(triage('Signed 2017','Signed 2018')['numeric_anchor_mismatch_count'],1)
        self.assertTrue(triage('', 'word '*161)['over_160_words'])

    def test_ordinals_and_anniversary(self):
        self.assertEqual(durations('ending on the second anniversary'),{(2,'','year')})
        self.assertEqual(triage('18th day','18')['numeric_anchor_mismatch_count'],0)
