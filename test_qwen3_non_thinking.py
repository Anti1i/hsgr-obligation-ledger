import unittest

from asqa_missing_selector_p6x import ModelRunner


class DummyTokenizer:
    def __init__(self):
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return "rendered"


class Qwen3NonThinkingTests(unittest.TestCase):
    def runner(self, model_id):
        runner = ModelRunner.__new__(ModelRunner)
        runner.model_id = model_id
        runner.chat_template_kwargs = (
            {"enable_thinking": False} if model_id.startswith("Qwen/Qwen3") else {}
        )
        runner.tokenizer = DummyTokenizer()
        return runner

    def test_qwen3_disables_thinking(self):
        runner = self.runner("Qwen/Qwen3-8B")
        self.assertEqual(runner.chat_text("x"), "rendered")
        self.assertIs(runner.tokenizer.kwargs["enable_thinking"], False)

    def test_qwen2_does_not_receive_unknown_keyword(self):
        runner = self.runner("Qwen/Qwen2.5-7B-Instruct")
        runner.chat_text("x")
        self.assertNotIn("enable_thinking", runner.tokenizer.kwargs)


if __name__ == "__main__":
    unittest.main()
