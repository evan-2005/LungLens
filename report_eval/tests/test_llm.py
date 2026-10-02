"""Tests for the LLM backends and the S2/S3/S4 generators, using stand-in models."""
import json
import os
import sys
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from report_eval.descriptors import NO_REGION, Descriptors  # noqa: E402
from report_eval.llm_backends import (  # noqa: E402
    BackendError,
    ClaudeBackend,
    LLMResult,
    OllamaBackend,
)
from report_eval.llm_generators import s2_llm, s3_vlm, s4_vlm  # noqa: E402
from report_eval.templates import Prediction  # noqa: E402

PRED = Prediction("Covid-19", 0.99, {"Normal": 0.0, "Pneumonia": 0.005, "Tuberculosis": 0.005,
                                     "Covid-19": 0.99}, "segmentation")
LEFT_MID = Descriptors(True, "left lung", "mid", "patchy", "multifocal", 0.8, False)


class FakeBackend:
    """Returns canned text and records what it was asked."""
    name = "fake"
    model = "fake-1"

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def generate(self, system, user, images=None, schema=None, max_tokens=1024):
        self.calls.append({"system": system, "user": user, "images": images or [], "schema": schema})
        text = self.reply if isinstance(self.reply, str) else json.dumps(self.reply)
        return LLMResult(text=text, model=self.model, latency_s=0.01)


def _s2_reply(**kw):
    base = {"state_location": True, "side": "left lung", "zone": "mid", "extent": "patchy",
            "sentence": "The heatmap highlights a patchy region in the left lung mid zone."}
    base.update(kw)
    return base


class S2Test(unittest.TestCase):
    def test_faithful_reply_is_accepted_and_claim_parsed(self):
        out = s2_llm(PRED, LEFT_MID, FakeBackend(_s2_reply()))
        self.assertFalse(out.rejected)
        self.assertEqual(out.claim.side, "left lung")
        self.assertIn("left lung mid zone", out.text)

    def test_schema_and_fields_are_sent(self):
        fake = FakeBackend(_s2_reply())
        s2_llm(PRED, LEFT_MID, fake)
        sent = json.loads(fake.calls[0]["user"])
        self.assertEqual(sent["side"], "left lung")
        self.assertTrue(sent["gate_pass"])
        self.assertIn("state_location", fake.calls[0]["schema"]["properties"])

    def test_changed_field_is_rejected_and_falls_back_to_template(self):
        out = s2_llm(PRED, LEFT_MID, FakeBackend(_s2_reply(side="right lung")))
        self.assertTrue(out.rejected)
        self.assertIn("side", out.reject_reason)
        self.assertIn("Heatmap evidence: left lung, mid zone", out.text)

    def test_location_invented_in_prose_is_rejected(self):
        reply = _s2_reply(sentence="Patchy change in the left mid zone and the right apex.")
        out = s2_llm(PRED, LEFT_MID, FakeBackend(reply))
        self.assertTrue(out.rejected)
        self.assertIn("sentence", out.reject_reason)

    def test_sentence_asserting_a_finding_is_rejected(self):
        reply = _s2_reply(sentence="The X-ray shows a patchy abnormality in the left lung mid zone.")
        out = s2_llm(PRED, LEFT_MID, FakeBackend(reply))
        self.assertTrue(out.rejected)
        self.assertIn("finding", out.reject_reason)

    def test_location_when_gate_blocks_is_rejected(self):
        out = s2_llm(PRED, NO_REGION, FakeBackend(_s2_reply()))
        self.assertTrue(out.rejected)
        self.assertIsNone(out.claim)

    def test_silence_when_gate_blocks_is_accepted(self):
        reply = {"state_location": False, "side": "none", "zone": "none", "extent": "none",
                 "sentence": "No heatmap region is reliable enough to report a location."}
        out = s2_llm(PRED, NO_REGION, FakeBackend(reply))
        self.assertFalse(out.rejected)
        self.assertIsNone(out.claim)

    def test_invalid_json_is_rejected_not_raised(self):
        out = s2_llm(PRED, LEFT_MID, FakeBackend("not json"))
        self.assertTrue(out.rejected)
        self.assertIn("json", out.reject_reason)


class VLMTest(unittest.TestCase):
    def test_s3_sends_one_image_and_parses_free_text(self):
        fake = FakeBackend("Ground-glass opacity in the right lower zone.")
        out = s3_vlm(PRED, b"png-bytes", fake)
        self.assertEqual(len(fake.calls[0]["images"]), 1)
        self.assertEqual(out.claim.side, "right lung")
        self.assertIsNone(fake.calls[0]["schema"])

    def test_s4i_asks_for_image_side_and_converts_to_patient_side(self):
        from report_eval.llm_generators import s4i_vlm
        fake = FakeBackend("The highlighted region is in the upper left of the image.")
        out = s4i_vlm(PRED, b"orig", b"overlay", fake)
        self.assertIn("image", fake.calls[0]["user"].lower())
        self.assertNotIn("patient's right", fake.calls[0]["system"])
        self.assertEqual(out.claim.side, "right lung")  # image-left = patient's right
        self.assertEqual(out.claim.zone, "upper")

    def test_s4_sends_radiograph_and_overlay(self):
        fake = FakeBackend("No clear focal abnormality.")
        out = s4_vlm(PRED, b"orig", b"overlay", fake)
        self.assertEqual(fake.calls[0]["images"], [b"orig", b"overlay"])
        self.assertIn("heatmap", fake.calls[0]["user"].lower())
        self.assertIsNone(out.claim)


class _FakeMessages:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def _claude_client(response):
    messages = _FakeMessages(response)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


class ClaudeBackendTest(unittest.TestCase):
    def _response(self, text="ok", stop_reason="end_turn"):
        return SimpleNamespace(content=[SimpleNamespace(type="thinking", thinking=""),
                                        SimpleNamespace(type="text", text=text)],
                               stop_reason=stop_reason, stop_details=None, model="claude-opus-5")

    def test_builds_request_with_images_schema_and_fallbacks(self):
        client, messages = _claude_client(self._response())
        backend = ClaudeBackend(client=client)
        result = backend.generate("sys", "hello", images=[b"\x89PNG"], schema={"type": "object"})
        kw = messages.kwargs
        self.assertEqual(kw["model"], "claude-opus-5")
        self.assertEqual(kw["fallbacks"], "default")
        self.assertIn("server-side-fallback-2026-07-01", kw["betas"])
        self.assertEqual(kw["output_config"]["format"]["type"], "json_schema")
        content = kw["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "image")
        self.assertEqual(content[-1], {"type": "text", "text": "hello"})
        self.assertEqual(result.text, "ok")

    def test_fallbacks_can_be_disabled(self):
        client, messages = _claude_client(self._response())
        ClaudeBackend(client=client, fallbacks=False).generate("sys", "hi")
        self.assertNotIn("fallbacks", messages.kwargs)
        self.assertNotIn("betas", messages.kwargs)

    def test_refusal_raises(self):
        client, _ = _claude_client(self._response(stop_reason="refusal"))
        with self.assertRaises(BackendError):
            ClaudeBackend(client=client).generate("sys", "hi")


class OllamaBackendTest(unittest.TestCase):
    def test_request_payload(self):
        captured = {}

        def fake_post(url, payload, timeout):
            captured.update(url=url, payload=payload)
            return {"message": {"content": "hi"}, "model": "m"}

        backend = OllamaBackend("m", post=fake_post)
        result = backend.generate("sys", "user", images=[b"abc"], schema={"type": "object"})
        p = captured["payload"]
        self.assertTrue(captured["url"].endswith("/api/chat"))
        self.assertEqual(p["messages"][1]["images"], ["YWJj"])
        self.assertEqual(p["format"], {"type": "object"})
        self.assertEqual(p["options"]["temperature"], 0)
        self.assertEqual(result.text, "hi")

    def test_unreachable_server_raises_backend_error(self):
        def failing_post(url, payload, timeout):
            raise OSError("connection refused")

        with self.assertRaises(BackendError):
            OllamaBackend("m", post=failing_post).generate("s", "u")


if __name__ == "__main__":
    unittest.main()
