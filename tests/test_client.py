import unittest

from agentic_latency.client import extract_sse_content


class ClientTest(unittest.TestCase):
    def test_sse_content_is_extracted_until_done(self) -> None:
        lines = [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}',
            b'data: {"choices":[{"delta":{"content":" world"}}]}',
            b"data: [DONE]",
            b'data: {"choices":[{"delta":{"content":"ignored"}}]}',
        ]

        self.assertEqual(list(extract_sse_content(lines)), ["Hello", " world"])


if __name__ == "__main__":
    unittest.main()

