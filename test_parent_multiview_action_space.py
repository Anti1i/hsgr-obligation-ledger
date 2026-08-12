import unittest

from parent_multiview_action_space import arm_prompt, stable_half


class ParentMultiViewActionSpaceTest(unittest.TestCase):
    def test_prompts_keep_question_and_check_is_label_free(self):
        unit = {"question": "What is 1+1?", "proposal": "3"}
        self.assertIn("What is 1+1?", arm_prompt(unit, "equation"))
        check = arm_prompt(unit, "check")
        self.assertIn("Previous proposed answer: 3", check)
        self.assertNotIn("correct answer", check.lower())

    def test_hash_half_is_stable(self):
        self.assertEqual(stable_half(17), stable_half(17))
        self.assertIn(stable_half(17), (0, 1))


if __name__ == "__main__":
    unittest.main()
