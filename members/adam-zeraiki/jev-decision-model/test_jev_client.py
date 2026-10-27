"""Focused offline tests; no API credentials or network requests are used."""
import copy
import decimal
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from jev_client import (
    APIRequestFailed, BudgetExceeded, BudgetedJevClient, ClientBusy, HTTPResult,
    InvalidRequest, InvalidResponse, LedgerError, MAX_INPUT_TOKENS, MissingAPIKey,
    MODEL, PRICE_PER_MILLION, PriorAttemptUnavailable, RESERVATION_USD,
    parse_choice, payload_sha256,
)


def payload(sample="a"):
    return {
        "model": MODEL,
        "state": {"sample": sample},
        "questions": {"provider": {
            "type": "choice", "instructions": "Select the image provider.",
            "criteria": {"apple": "Apple emoji style", "google": "Google emoji style"},
        }},
    }


def response(tokens=1000):
    return {
        "model": MODEL,
        "answers": {"provider": {
            "type": "choice", "choice": "apple",
            "probabilities": {"apple": 0.8, "google": 0.2}, "confidence": 0.6,
        }},
        "usage": {"input_tokens": tokens, "output_tokens": 34},
    }


def http_result(value=None, status=200):
    return HTTPResult(status, json.dumps(response() if value is None else value).encode())


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_success_uses_reported_input_and_free_outputs(self):
        client = BudgetedJevClient(self.root, transport=lambda body: http_result())
        result = client.run(payload(), "trial:a", {"mode": "third_expert"})
        self.assertEqual(parse_choice(result, "provider", ["apple", "google"])["choice"], "apple")
        summary = client.budget_summary()
        self.assertEqual(decimal.Decimal(summary["known_usage_cost_usd"]), decimal.Decimal("0.000042"))
        self.assertEqual(decimal.Decimal(summary["unresolved_reservations_usd"]), 0)
        self.assertEqual(summary["attempts"], 1)

    def test_cache_reuse_is_offline_and_does_not_charge(self):
        calls = []
        client = BudgetedJevClient(self.root, transport=lambda body: (calls.append(body) or http_result()))
        first = client.run(payload())
        with mock.patch.dict(os.environ, {}, clear=True):
            second = BudgetedJevClient(self.root).run(payload(), "second-key")
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)
        self.assertEqual(client.budget_summary()["attempts"], 1)

    def test_hard_cap_reserves_before_call_and_refuses_next(self):
        calls = []
        cap = str(RESERVATION_USD)
        client = BudgetedJevClient(
            self.root, cap,
            transport=lambda body: (calls.append(body) or http_result(response(MAX_INPUT_TOKENS))),
        )
        client.run(payload("first"))
        with self.assertRaises(BudgetExceeded):
            client.run(payload("second"))
        self.assertEqual(len(calls), 1)
        self.assertEqual(decimal.Decimal(client.budget_summary()["committed_upper_bound_usd"]), RESERVATION_USD)

    def test_ambiguous_attempt_stays_reserved_and_cannot_retry(self):
        calls = []
        def fail(body):
            calls.append(body)
            raise TimeoutError("transport text should not be stored")
        client = BudgetedJevClient(self.root, transport=fail)
        with self.assertRaises(APIRequestFailed):
            client.run(payload())
        with self.assertRaises(PriorAttemptUnavailable):
            client.run(payload())
        summary = client.budget_summary()
        self.assertEqual(decimal.Decimal(summary["unresolved_reservations_usd"]), RESERVATION_USD)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("transport text", (self.root / "budget_ledger.json").read_text())

    def test_invalid_json_stays_reserved(self):
        client = BudgetedJevClient(self.root, transport=lambda body: HTTPResult(200, b"not json"))
        with self.assertRaises(InvalidResponse):
            client.run(payload())
        self.assertEqual(decimal.Decimal(client.budget_summary()["unresolved_reservations_usd"]), RESERVATION_USD)

    def test_malformed_answer_retains_valid_reported_charge(self):
        malformed = response(5000)
        malformed["answers"]["provider"]["probabilities"] = {"apple": 0.9, "google": 0.9}
        client = BudgetedJevClient(self.root, transport=lambda body: http_result(malformed))
        with self.assertRaises(InvalidResponse):
            client.run(payload())
        summary = client.budget_summary()
        self.assertEqual(decimal.Decimal(summary["known_usage_cost_usd"]), decimal.Decimal("0.00021"))
        self.assertEqual(decimal.Decimal(summary["unresolved_reservations_usd"]), 0)
        with self.assertRaises(PriorAttemptUnavailable):
            client.run(payload())

    def test_http_error_with_unknown_usage_keeps_reservation(self):
        client = BudgetedJevClient(self.root, transport=lambda body: http_result({"error": "rate limited"}, 429))
        with self.assertRaises(APIRequestFailed):
            client.run(payload())
        self.assertEqual(decimal.Decimal(client.budget_summary()["unresolved_reservations_usd"]), RESERVATION_USD)

    def test_missing_key_does_not_reserve(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            client = BudgetedJevClient(self.root)
            with self.assertRaises(MissingAPIKey):
                client.run(payload())
            self.assertEqual(client.budget_summary()["attempts"], 0)

    def test_concurrent_client_is_rejected_before_second_attempt(self):
        entered = threading.Event()
        finish = threading.Event()
        results = []
        errors = []
        def transport(body):
            entered.set()
            self.assertTrue(finish.wait(5))
            return http_result()
        first = BudgetedJevClient(self.root, transport=transport)
        second = BudgetedJevClient(self.root, transport=lambda body: http_result())
        def run_first():
            try:
                results.append(first.run(payload()))
            except BaseException as exc:
                errors.append(exc)
        worker = threading.Thread(target=run_first)
        worker.start()
        self.assertTrue(entered.wait(5))
        try:
            with self.assertRaises(ClientBusy):
                second.run(payload())
        finally:
            finish.set()
            worker.join(5)
        self.assertFalse(errors)
        self.assertEqual(len(results), 1)
        self.assertEqual(first.budget_summary()["attempts"], 1)

    def test_interrupted_reserved_entry_blocks_replay(self):
        def interrupt(body):
            raise KeyboardInterrupt
        client = BudgetedJevClient(self.root, transport=interrupt)
        with self.assertRaises(KeyboardInterrupt):
            client.run(payload())
        with self.assertRaises(PriorAttemptUnavailable):
            BudgetedJevClient(self.root, transport=lambda body: http_result()).run(payload())
        self.assertEqual(decimal.Decimal(client.budget_summary()["unresolved_reservations_usd"]), RESERVATION_USD)

    def test_probability_cache_mutation_is_rejected(self):
        client = BudgetedJevClient(self.root, transport=lambda body: http_result())
        client.run(payload())
        path = self.root / "calls" / payload_sha256(payload()) / "response.json"
        modified = json.loads(path.read_text())
        modified["answers"]["provider"].update(choice="google", probabilities={"apple": 0.2, "google": 0.8})
        path.write_text(json.dumps(modified))
        with self.assertRaises(LedgerError):
            client.run(payload())
        self.assertEqual(client.budget_summary()["attempts"], 1)

    def test_raw_cache_mutation_is_rejected(self):
        client = BudgetedJevClient(self.root, transport=lambda body: http_result())
        client.run(payload())
        path = self.root / "calls" / payload_sha256(payload()) / "response.raw"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaises(LedgerError):
            client.run(payload())

    def test_key_rebinding_and_model_alias_are_rejected(self):
        client = BudgetedJevClient(self.root, transport=lambda body: http_result())
        client.run(payload(), "one-key")
        with self.assertRaises(InvalidRequest):
            client.run(payload("changed"), "one-key")
        alias = payload()
        alias["model"] = "jev-latest"
        with self.assertRaises(InvalidRequest):
            client.run(alias)
        self.assertEqual(client.budget_summary()["attempts"], 1)

    def test_provider_context_overrun_freezes_future_calls(self):
        client = BudgetedJevClient(self.root, transport=lambda body: http_result(response(MAX_INPUT_TOKENS + 1)))
        with self.assertRaises(InvalidResponse):
            client.run(payload())
        self.assertTrue(client.budget_summary()["frozen"])
        with self.assertRaises(BudgetExceeded):
            client.run(payload("second"))

    def test_bounded_hundredth_grid_rounding_normalizes_099_and_102(self):
        classes = ["apple", "facebook", "google", "messenger", "mozilla", "samsung", "whatsapp"]
        for vector, expected_sum in (
            ([0.28, 0.01, 0.03, 0.0, 0.0, 0.0, 0.67], 0.99),
            ([0.28, 0.01, 0.03, 0.01, 0.0, 0.01, 0.68], 1.02),
        ):
            with self.subTest(expected_sum=expected_sum):
                raw = response()
                raw["answers"]["provider"].update(choice="whatsapp", probabilities=dict(zip(classes, vector)))
                original = copy.deepcopy(raw)
                parsed = parse_choice(raw, "provider", classes)
                self.assertAlmostEqual(parsed["raw_probability_sum"], expected_sum)
                self.assertTrue(parsed["normalization_applied"])
                self.assertEqual(parsed["normalization_reason"], "bounded_hundredth_grid_rounding")
                self.assertAlmostEqual(sum(parsed["probabilities"].values()), 1.0)
                self.assertAlmostEqual(parsed["probabilities"]["whatsapp"], vector[-1] / expected_sum)
                self.assertEqual(raw, original)

    def test_rounding_adapter_rejects_non_grid_and_out_of_bound_vectors(self):
        classes = ["apple", "facebook", "google", "messenger", "mozilla", "samsung", "whatsapp"]
        for vector in (
            [0.281, 0.009, 0.03, 0.0, 0.0, 0.0, 0.67],  # Sum .99, not hundredths.
            [0.28, 0.01, 0.03, 0.01, 0.0, 0.01, 0.70],  # Sum 1.04 exceeds .035 bound.
            [0.0] * 7,
            [0.28, -0.01, 0.03, 0.0, 0.0, 0.0, 0.69],
        ):
            with self.subTest(vector=vector):
                raw = response()
                raw["answers"]["provider"].update(choice="whatsapp", probabilities=dict(zip(classes, vector)))
                with self.assertRaises(InvalidResponse):
                    parse_choice(raw, "provider", classes)

    def test_credential_echo_is_redacted_including_json_escapes(self):
        # Fake credential is only an offline test sentinel, never a real key.
        key = "TEST_CREDENTIAL_DO_NOT_PERSIST"
        escaped = "".join("\\u%04x" % ord(c) for c in key)
        raw = ('{"error":"' + escaped + '"}').encode()
        with mock.patch.dict(os.environ, {"TYPESAFE_API_KEY": key}):
            with mock.patch("jev_client._network_once", return_value=HTTPResult(401, raw)):
                client = BudgetedJevClient(self.root)
                with self.assertRaises(APIRequestFailed):
                    client.run(payload())
        for path in self.root.rglob("*"):
            if path.is_file():
                self.assertNotIn(key.encode(), path.read_bytes())
                self.assertNotIn(escaped.encode(), path.read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
