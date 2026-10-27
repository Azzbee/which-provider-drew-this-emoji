"""Offline semantic tests. Synthetic responses never enter the real prepared run."""
import argparse
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

import jev_pipeline as p


CLIENT = Path(__file__).with_name("jev_client.py")
if not CLIENT.exists():
    CLIENT = Path(__file__).parent.parent / "client_work" / "jev_client.py"


class PipelineSemantics(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="pipeline_test_", dir=Path(__file__).parent)
        self.root = Path(self.tmp.name)
        img = self.root / "image.png"
        Image.new("RGB", (280, 280), "white").save(img)
        sha = p.sha256(img)
        rows = [{"id": f"{i+1:05}", "label": int(i >= 8), "group": i,
                 "row": i, "image": "image.png", "prepared_sha256": sha} for i in range(12)]
        dino = np.zeros((12, 7)); conv = np.zeros((12, 7))
        dino[:8, :2] = [.9, .1]; conv[:8, :2] = [.8, .2]
        dino[8:, :2] = [.4, .6]; conv[8:, :2] = [.7, .3]
        np.save(self.root / "dino.npy", dino); np.save(self.root / "conv.npy", conv)
        self.protocol = {
            "model": p.MODEL, "class_order": p.CLASSES, "pipeline_sha256": p.sha256(p.__file__),
            "prompts": {"expert": p.EXPERT_INSTRUCTIONS, "final": p.FINAL_INSTRUCTIONS},
            "references": [{"class": c, "image": "image.png", "prepared_sha256": sha} for c in p.CLASSES],
            "partitions": {"development": {"rows": rows, "dino_file": "dino.npy", "convnext_file": "conv.npy",
                "dino_sha256": p.sha256(self.root / "dino.npy"), "convnext_sha256": p.sha256(self.root / "conv.npy")}},
            "gate": {"candidate_priority": ["third_expert_blend", "final_jev", "disagreement_router"]},
            "evaluation_warning": "SYNTHETIC TEST ONLY"}
        p.write_json(self.root / "protocol.json", self.protocol)
        (self.root / "protocol.sha256").write_text(p.digest(self.protocol))
        self.rows, self.dino, self.conv = rows, dino, conv
        self.args = argparse.Namespace(run_dir=str(self.root), client_module=str(CLIENT))
        self.client_module = p.load_client_module(CLIENT)

    def tearDown(self):
        self.tmp.cleanup()

    def response(self, chosen=1):
        probabilities = {c: float(i == chosen) for i, c in enumerate(p.CLASSES)}
        return {"model": p.MODEL, "answers": {"provider": {"type": "choice",
                    "choice": p.CLASSES[chosen], "confidence": 1., "probabilities": probabilities}},
                "usage": {"input_tokens": 10, "output_tokens": 0}}

    def cache_synthetic(self, i):
        raw = self.response()
        j = np.array([0., 1., 0., 0., 0., 0., 0.])
        expert = p.payload_for(self.root, self.protocol, self.rows[i])
        final = p.payload_for(self.root, self.protocol, self.rows[i], self.dino[i], self.conv[i], j)
        item = {"protocol_sha256": p.digest(self.protocol), "partition": "development", "id": self.rows[i]["id"],
                "expert": {"payload_sha256": p.digest(expert), "response_sha256": p.digest(raw), "response": raw},
                "final": {"payload_sha256": p.digest(final), "response_sha256": p.digest(raw), "response": raw}}
        p.write_json(p.outcome_path(self.root, "development", self.rows[i]), item)

    def test_two_stage_inputs_have_no_query_label_or_id(self):
        row = dict(self.rows[0], id="SECRET_QUERY_ID", label="SECRET_TRUE_LABEL")
        expert = p.payload_for(self.root, self.protocol, row)
        self.assertNotIn("expert_probabilities", expert["state"])
        text = p.canonical(expert).decode()
        self.assertNotIn("SECRET_QUERY_ID", text); self.assertNotIn("SECRET_TRUE_LABEL", text)
        j = np.full(7, 1/7)
        final = p.payload_for(self.root, self.protocol, row, self.dino[0], self.conv[0], j)
        self.assertEqual(set(final["state"]["expert_probabilities"]), {"dinov2", "convnext", "jev_image_only"})
        self.assertAlmostEqual(final["state"]["expert_probabilities"]["jev_image_only"]["apple"], 1/7)
        self.client_module._validate_payload(expert)
        self.client_module._validate_payload(final)

    def test_partial_router_cannot_report_candidate(self):
        for i in range(8, 11):
            self.cache_synthetic(i)
        report = p.evaluate(self.args)
        self.assertEqual(report["candidates"], {})
        self.assertEqual(report["routed_rows_completed"], 3)
        self.assertIsNone(report["selected_deployment_candidate"])

    def test_complete_router_uses_unchanged_baseline_elsewhere(self):
        for i in range(8, 12):
            self.cache_synthetic(i)
        report = p.evaluate(self.args)
        self.assertEqual(set(report["candidates"]), {"disagreement_router"})
        self.assertEqual(report["baseline"]["correct"], 8)
        self.assertEqual(report["candidates"]["disagreement_router"]["correct"], 12)
        self.assertEqual(report["candidates"]["disagreement_router"]["net_additional_correct"], 4)
        self.assertFalse(report["candidates"]["disagreement_router"]["eligible_for_deployment"])

    def test_full_candidates_only_with_all_results(self):
        for i in range(12):
            self.cache_synthetic(i)
        report = p.evaluate(self.args)
        self.assertEqual(report["complete_two_stage_rows"], 12)
        self.assertEqual(report["candidates"]["final_jev"]["correct"], 4)
        self.assertEqual(report["candidates"]["third_expert_blend"]["correct"], 12)
        self.assertEqual(report["candidates"]["disagreement_router"]["correct"], 12)

    def test_response_tampering_is_rejected(self):
        self.cache_synthetic(8)
        path = p.outcome_path(self.root, "development", self.rows[8])
        item = p.read_json(path)
        item["final"]["response"]["model"] = "not-jev"
        p.write_json(path, item)
        with self.assertRaisesRegex(ValueError, "response hash mismatch"):
            p.evaluate(self.args)

    def test_pipeline_mutation_blocks_reuse(self):
        self.protocol["pipeline_sha256"] = "0" * 64
        p.write_json(self.root / "protocol.json", self.protocol)
        (self.root / "protocol.sha256").write_text(p.digest(self.protocol))
        with self.assertRaisesRegex(ValueError, "Pipeline code changed"):
            p.evaluate(self.args)


if __name__ == "__main__":
    unittest.main()
