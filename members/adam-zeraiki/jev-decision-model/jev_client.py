#!/usr/bin/env python3
"""Auditable, single-attempt TypeSafe Jev image client for a capped experiment.

The client deliberately has no retry policy. It reserves the cost of 65,536 input
tokens before each network attempt, holds one cross-process lock throughout the
attempt, and permanently retains that reservation when billing is unknown. All
modes MUST use the same cache_dir. A completed payload can be replayed offline.

Official references, checked 2026-10-18:
  https://docs.typesafe.ai/api
  https://docs.typesafe.ai/images
  https://docs.typesafe.ai/models

Only TYPESAFE_API_KEY supplies authentication. No endpoint, headers, model alias,
long-context opt-in, automatic retry, or API-key command-line option is exposed.
The published 64k limit is rounded conservatively to 65,536 tokens. The spending
bound depends on the provider respecting its documented context and price.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import datetime as dt
import decimal
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import ssl
import urllib.error
import urllib.request
from typing import Any, Callable, Iterable, Mapping, NamedTuple


MODEL = "jev-1.14.0"
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
API_HOST = "api.typesafe.ai"
API_PATH = "/v1/systemone"
AUTH_ENV = "TYPESAFE_API_KEY"
PRICE_PER_MILLION = decimal.Decimal("0.042")
MAX_INPUT_TOKENS = 65_536
MAX_AUTHORIZED_USD = decimal.Decimal("5")
RESERVATION_USD = PRICE_PER_MILLION * MAX_INPUT_TOKENS / 1_000_000
MAX_BODY_BYTES = 32 * 1024 * 1024
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
REQUEST_KEY_RE = re.compile(r"[A-Za-z0-9._:-]{1,200}\Z")


class JevClientError(RuntimeError):
    """Safe-to-display client error: messages never contain request credentials."""


class MissingAPIKey(JevClientError):
    pass


class BudgetExceeded(JevClientError):
    pass


class ClientBusy(JevClientError):
    pass


class PriorAttemptUnavailable(JevClientError):
    pass


class InvalidRequest(JevClientError):
    pass


class InvalidResponse(JevClientError):
    pass


class APIRequestFailed(JevClientError):
    pass


class LedgerError(JevClientError):
    pass


class HTTPResult(NamedTuple):
    status: int
    body: bytes
    headers: Mapping[str, str] = {}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _decimal_text(value: decimal.Decimal) -> str:
    return format(value, "f")


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise InvalidRequest("Request must contain finite, UTF-8 JSON values.") from None


def payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        # Persist the rename as well as the file contents before the HTTP call.
        dir_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, _canonical(value) + b"\n")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        raise LedgerError("An experiment cache file is missing or invalid; no API call was made.") from None


def _exact_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def parse_choice(
    response: Mapping[str, Any], question_name: str, class_order: Iterable[str],
) -> dict[str, Any]:
    """Validate Choice output and adapt bounded hundredth-grid API rounding.

    General vectors retain the original 1e-5 sum tolerance. When every value is
    within 1e-8 of a hundredth, a larger discrepancy is accepted only within
    K * 0.005 + 1e-8, the bound for independent rounding of K probabilities.
    Such vectors are divided by their positive sum. Raw API responses remain
    unchanged; the returned fields record the original sum and normalization.
    """
    classes = tuple(class_order)
    if not classes or len(set(classes)) != len(classes):
        raise InvalidResponse("Expected class order is empty or contains duplicates.")
    if not isinstance(response, Mapping) or response.get("model") != MODEL:
        raise InvalidResponse("Response model does not match pinned jev-1.14.0.")
    answers = response.get("answers")
    answer = answers.get(question_name) if isinstance(answers, Mapping) else None
    if not isinstance(answer, Mapping) or answer.get("type") != "choice":
        raise InvalidResponse("Response is missing a requested Choice answer.")
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, Mapping) or set(probabilities) != set(classes):
        raise InvalidResponse("Choice probabilities do not match the expected classes.")
    if any(not _exact_number(p) or not 0 <= p <= 1 for p in probabilities.values()):
        raise InvalidResponse("Choice probabilities must be finite numbers between zero and one.")
    raw_sum = math.fsum(probabilities.values())
    discrepancy = abs(raw_sum - 1.0)
    hundredth_grid = all(abs(p - round(p, 2)) <= 1e-8 for p in probabilities.values())
    normalization_applied = False
    if discrepancy > 1e-5:
        if raw_sum <= 0 or not hundredth_grid or discrepancy > len(classes) * 0.005 + 1e-8:
            raise InvalidResponse("Choice probabilities do not sum to one within the bounded API rounding tolerance.")
        normalization_applied = True
    choice = answer.get("choice")
    if choice not in probabilities:
        raise InvalidResponse("Choice is not an expected class.")
    if probabilities[choice] + 1e-8 < max(probabilities.values()):
        raise InvalidResponse("Chosen class is not a highest-probability option.")
    confidence = answer.get("confidence")
    if not _exact_number(confidence) or not 0 <= confidence <= 1:
        raise InvalidResponse("Choice confidence must be a finite number between zero and one.")
    return {
        "model": MODEL,
        "choice": choice,
        "probabilities": {
            label: float(probabilities[label] / raw_sum if normalization_applied else probabilities[label])
            for label in classes
        },
        "confidence": float(confidence),
        "raw_probability_sum": float(raw_sum),
        "normalization_applied": normalization_applied,
        "normalization_reason": "bounded_hundredth_grid_rounding" if normalization_applied else None,
    }


def _usage(response: Any) -> tuple[int, int] | None:
    if not isinstance(response, dict) or not isinstance(response.get("usage"), dict):
        return None
    usage = response["usage"]
    input_tokens, output_tokens = usage.get("input_tokens"), usage.get("output_tokens")
    if type(input_tokens) is not int or type(output_tokens) is not int:
        return None
    if input_tokens < 0 or output_tokens < 0:
        return None
    return input_tokens, output_tokens


def _validate_payload(payload: Mapping[str, Any]) -> bytes:
    if not isinstance(payload, Mapping) or set(payload) - {"model", "state", "questions", "assets"}:
        raise InvalidRequest("Only model, state, questions and assets are allowed in this experiment.")
    if payload.get("model") != MODEL:
        raise InvalidRequest("This experiment requires the exact model jev-1.14.0.")
    if "state" not in payload:
        raise InvalidRequest("Request state is required.")
    questions = payload.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise InvalidRequest("At least one named Choice question is required.")
    for key, question in questions.items():
        if not isinstance(key, str) or not key:
            raise InvalidRequest("Question names must be nonempty strings.")
        if not isinstance(question, dict) or set(question) != {"type", "instructions", "criteria"}:
            raise InvalidRequest("Each question must contain only type, instructions and criteria.")
        if question["type"] != "choice":
            raise InvalidRequest("This experiment accepts Choice questions only.")
        criteria = question["criteria"]
        if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
            raise InvalidRequest("A Choice question needs between two and 255 options.")
        if any(not isinstance(label, str) or not label for label in criteria):
            raise InvalidRequest("Choice option names must be nonempty strings.")
    assets = payload.get("assets", {})
    if not isinstance(assets, dict):
        raise InvalidRequest("Assets must be an object.")
    for key, asset in assets.items():
        if not isinstance(key, str) or not key.strip() or len(key.encode("utf-8")) > 1024:
            raise InvalidRequest("Image asset references must be nonempty strings of at most 1,024 UTF-8 bytes.")
        if not isinstance(asset, dict) or set(asset) != {"type", "format", "data_base64"}:
            raise InvalidRequest("Images need exactly type, format and data_base64 fields.")
        if asset["type"] != "image" or asset["format"] not in {"png", "jpg", "jpeg"}:
            raise InvalidRequest("Only PNG and JPEG image assets are supported.")
        try:
            data = base64.b64decode(asset["data_base64"], validate=True)
        except (TypeError, ValueError):
            raise InvalidRequest("An image asset is not valid base64.") from None
        if not data or len(data) > 16 * 1024 * 1024:
            raise InvalidRequest("Image assets must be nonempty and at most 16 MiB decoded.")
    body = _canonical(payload)
    if len(body) > MAX_BODY_BYTES:
        raise InvalidRequest("Request JSON exceeds the 32 MiB API limit.")
    return body


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never resend a credential-bearing request to a redirected endpoint."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _network_once(body: bytes, api_key: str, timeout: float) -> HTTPResult:
    """One fixed HTTPS POST using the runtime's authorized proxy and CA bundle.

    urllib's ProxyHandler honors HTTPS_PROXY/NO_PROXY. Neither redirects nor
    automatic retries are enabled. HTTP errors remain ordinary responses for
    the caller's conservative usage and reservation accounting.
    """
    cafile = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE") or None
    tls_context = ssl.create_default_context(cafile=cafile)
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler(),
        urllib.request.HTTPSHandler(context=tls_context),
        _NoRedirect(),
    )
    request = urllib.request.Request(
        ENDPOINT, data=body, method="POST",
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "TypeSafe-Danger-Zone-Images": "true",
            "User-Agent": "ESSEC-emoji-jev-experiment/1.0",
        },
    )
    try:
        response = opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        data = response.read(MAX_RESPONSE_BYTES + 1)
        if len(data) > MAX_RESPONSE_BYTES:
            raise InvalidResponse("API response exceeds the local safety size limit.")
        safe_headers = {
            key.lower(): value for key, value in response.headers.items()
            if key.lower() in {"x-request-id", "request-id", "retry-after", "content-type"}
        }
        return HTTPResult(response.status, data, safe_headers)


class BudgetedJevClient:
    def __init__(
        self, cache_dir: str | Path, budget_usd: str = "5", *,
        timeout: float = 45.0,
        transport: Callable[[bytes], HTTPResult] | None = None,
    ) -> None:
        """All clients/modes in one experiment must share this cache directory.

        ``transport`` is dependency injection for offline tests. It receives only
        the body, never credentials; do not supply it for a live experiment.
        """
        try:
            self.cap = decimal.Decimal(str(budget_usd))
        except decimal.InvalidOperation:
            raise InvalidRequest("Budget must be a positive USD amount of at most $5.") from None
        if not self.cap.is_finite() or not 0 < self.cap <= MAX_AUTHORIZED_USD:
            raise InvalidRequest("Budget must be a positive USD amount of at most $5.")
        if not _exact_number(timeout) or not 0 < timeout <= 60:
            raise InvalidRequest("HTTP timeout must be greater than zero and at most 60 seconds.")
        self.root = Path(cache_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.ledger_path = self.root / "budget_ledger.json"
        self.lock_path = self.root / "budget.lock"
        self.timeout = float(timeout)
        self.transport = transport

    @contextlib.contextmanager
    def _lock(self):
        descriptor = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ClientBusy("Another process is using the experiment budget; no call was made.") from None
            yield
        finally:
            os.close(descriptor)

    def _ledger(self) -> dict[str, Any]:
        if not self.ledger_path.exists():
            ledger = {
                "schema_version": 1,
                "created_at": _now(),
                "model": MODEL,
                "endpoint": ENDPOINT,
                "currency": "USD",
                "budget_usd": _decimal_text(self.cap),
                "input_price_per_million_usd": _decimal_text(PRICE_PER_MILLION),
                "max_input_tokens_per_attempt": MAX_INPUT_TOKENS,
                "reservation_usd": _decimal_text(RESERVATION_USD),
                "frozen": False,
                "attempts": {},
                "request_keys": {},
            }
            _atomic_json(self.ledger_path, ledger)
            return ledger
        ledger = _read_json(self.ledger_path)
        expected = {
            "schema_version": 1,
            "model": MODEL,
            "endpoint": ENDPOINT,
            "currency": "USD",
            "budget_usd": _decimal_text(self.cap),
            "input_price_per_million_usd": _decimal_text(PRICE_PER_MILLION),
            "max_input_tokens_per_attempt": MAX_INPUT_TOKENS,
            "reservation_usd": _decimal_text(RESERVATION_USD),
        }
        if not isinstance(ledger, dict) or any(ledger.get(k) != v for k, v in expected.items()):
            raise LedgerError("Existing budget configuration differs or is invalid; no call was made.")
        if not isinstance(ledger.get("attempts"), dict) or not isinstance(ledger.get("request_keys"), dict):
            raise LedgerError("Experiment budget ledger is malformed; no call was made.")
        self._totals(ledger)  # Detect malformed accounting before any network call.
        return ledger

    @staticmethod
    def _totals(ledger: dict[str, Any]) -> tuple[decimal.Decimal, decimal.Decimal]:
        known = decimal.Decimal(0)
        unresolved = decimal.Decimal(0)
        try:
            for key, entry in ledger["attempts"].items():
                if not re.fullmatch(r"[0-9a-f]{64}", key) or not isinstance(entry, dict):
                    raise ValueError
                actual = entry.get("actual_cost_usd")
                if actual is None:
                    if entry.get("reserved_usd") != _decimal_text(RESERVATION_USD):
                        raise ValueError
                    unresolved += RESERVATION_USD
                else:
                    amount = decimal.Decimal(actual)
                    if not amount.is_finite() or amount < 0:
                        raise ValueError
                    known += amount
        except (decimal.InvalidOperation, ValueError, TypeError, KeyError):
            raise LedgerError("Experiment budget accounting is invalid; no call was made.") from None
        return known, unresolved

    def budget_summary(self) -> dict[str, Any]:
        with self._lock():
            ledger = self._ledger()
            known, unresolved = self._totals(ledger)
            statuses: dict[str, int] = {}
            for entry in ledger["attempts"].values():
                status = entry["status"]
                statuses[status] = statuses.get(status, 0) + 1
            return {
                "budget_usd": _decimal_text(self.cap),
                "known_usage_cost_usd": _decimal_text(known),
                "unresolved_reservations_usd": _decimal_text(unresolved),
                "committed_upper_bound_usd": _decimal_text(known + unresolved),
                "remaining_uncommitted_usd": _decimal_text(max(decimal.Decimal(0), self.cap - known - unresolved)),
                "reservation_per_attempt_usd": _decimal_text(RESERVATION_USD),
                "attempts": len(ledger["attempts"]),
                "statuses": statuses,
                "frozen": bool(ledger.get("frozen")),
            }

    def run(
        self, payload: Mapping[str, Any], request_key: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = _validate_payload(payload)
        digest = hashlib.sha256(body).hexdigest()
        if request_key is not None and (not isinstance(request_key, str) or not REQUEST_KEY_RE.fullmatch(request_key)):
            raise InvalidRequest("Request key must be 1–200 letters, digits, dots, hyphens, underscores or colons.")
        provenance = {} if context is None else context
        if not isinstance(provenance, Mapping):
            raise InvalidRequest("Context must be a JSON object.")
        context_bytes = _canonical(provenance)
        if len(context_bytes) > 16 * 1024:
            raise InvalidRequest("Context exceeds the 16 KiB local provenance limit.")
        # Read the credential only to detect accidental inclusion and to send it.
        # No credential hash, prefix, value, headers, or exception repr is logged.
        api_key = os.environ.get(AUTH_ENV, "") if self.transport is None else ""
        if api_key and (
            api_key.encode() in body or api_key.encode() in context_bytes
            or (request_key is not None and api_key in request_key)
        ):
            raise InvalidRequest("A credential was included in request data; no data was saved or sent.")
        with self._lock():
            ledger = self._ledger()
            if request_key in ledger["request_keys"] and ledger["request_keys"][request_key] != digest:
                raise InvalidRequest("Request key was already bound to a different payload; no call was made.")
            entry = ledger["attempts"].get(digest)
            call_dir = self.root / "calls" / digest
            if entry is not None:
                if entry.get("status") != "complete":
                    raise PriorAttemptUnavailable(
                        "This payload already had an unsuccessful or interrupted attempt; it will not be retried."
                    )
                response = _read_json(call_dir / "response.json")
                try:
                    raw_cached = (call_dir / "response.raw").read_bytes()
                    original_response = json.loads(raw_cached.decode("utf-8"))
                    integrity_ok = (
                        hashlib.sha256(raw_cached).hexdigest() == entry.get("response_sha256")
                        and _canonical(response) == _canonical(original_response)
                        and _usage(response) == (entry.get("input_tokens"), entry.get("output_tokens"))
                        and entry.get("actual_cost_usd") == _decimal_text(
                            PRICE_PER_MILLION * response["usage"]["input_tokens"] / 1_000_000
                        )
                    )
                except (OSError, ValueError, UnicodeError, KeyError, TypeError, decimal.InvalidOperation):
                    integrity_ok = False
                if not integrity_ok:
                    raise LedgerError("Cached response integrity check failed; no API call was made.")
                self._validate_response(response, payload)
                if request_key is not None and request_key not in ledger["request_keys"]:
                    ledger["request_keys"][request_key] = digest
                    _atomic_json(self.ledger_path, ledger)
                return response
            if ledger.get("frozen"):
                raise BudgetExceeded("Budget ledger is frozen after a provider accounting anomaly; no call was made.")
            known, unresolved = self._totals(ledger)
            if known + unresolved + RESERVATION_USD > self.cap:
                raise BudgetExceeded("The remaining budget cannot cover one maximum-cost attempt; no call was made.")
            if self.transport is None and not api_key:
                raise MissingAPIKey("TYPESAFE_API_KEY is not configured. No request was sent and no budget was reserved.")
            if api_key and (api_key != api_key.strip() or any(ord(c) < 33 or ord(c) > 126 for c in api_key)):
                raise MissingAPIKey("TYPESAFE_API_KEY has an invalid format. No request was sent.")
            entry = {
                "payload_sha256": digest,
                "status": "reserved",
                "started_at": _now(),
                "request_key": request_key,
                "reserved_usd": _decimal_text(RESERVATION_USD),
                "actual_cost_usd": None,
                "input_tokens": None,
                "output_tokens": None,
                "context": provenance,
            }
            # Store request and ledger durably before the first network byte.
            _atomic_bytes(call_dir / "request.json", body + b"\n")
            ledger["attempts"][digest] = entry
            if request_key is not None:
                ledger["request_keys"][request_key] = digest
            _atomic_json(self.ledger_path, ledger)
            try:
                result = self.transport(body) if self.transport is not None else _network_once(body, api_key, self.timeout)
            except Exception:
                entry.update(status="ambiguous_transport_failure", finished_at=_now(), error_code="transport_exception")
                _atomic_json(call_dir / "record.json", entry)
                _atomic_json(self.ledger_path, ledger)
                raise APIRequestFailed("The API attempt failed or timed out. Its maximum-cost reservation remains; no retry was made.") from None
            try:
                status, raw, headers = result
                if type(status) is not int or not 100 <= status <= 599 or not isinstance(raw, bytes):
                    raise ValueError
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise ValueError
            except (TypeError, ValueError):
                entry.update(status="ambiguous_transport_failure", finished_at=_now(), error_code="invalid_transport_result")
                _atomic_json(call_dir / "record.json", entry)
                _atomic_json(self.ledger_path, ledger)
                raise APIRequestFailed("Transport did not return a valid HTTP response. Its reservation remains.") from None
            # Preserve raw API body unless it echoes the credential. Credential
            # redaction is explicit and stops acceptance of that response.
            redacted = bool(api_key and api_key.encode() in raw)
            stored_raw = raw.replace(api_key.encode(), b"[REDACTED_CREDENTIAL]") if redacted else raw
            # JSON escaping may hide an echoed ASCII key in the wire bytes.
            # Detect its decoded value before saving either raw or parsed output.
            if api_key:
                try:
                    decoded_for_redaction = json.loads(stored_raw.decode("utf-8"))
                    canonical_for_redaction = _canonical(decoded_for_redaction)
                    if api_key.encode() in canonical_for_redaction:
                        redacted = True
                        stored_raw = canonical_for_redaction.replace(api_key.encode(), b"[REDACTED_CREDENTIAL]")
                except (ValueError, UnicodeError, InvalidRequest):
                    pass
            safe_headers = {}
            if isinstance(headers, Mapping):
                for key, value in headers.items():
                    if str(key).lower() in {"x-request-id", "request-id", "retry-after", "content-type"}:
                        text = str(value)
                        safe_headers[str(key).lower()] = text.replace(api_key, "[REDACTED_CREDENTIAL]") if api_key else text
            _atomic_bytes(call_dir / "response.raw", stored_raw)
            entry.update(
                finished_at=_now(), http_status=status,
                response_sha256=hashlib.sha256(stored_raw).hexdigest(),
                credential_redaction_applied=redacted, response_headers=safe_headers,
            )
            try:
                response = json.loads(stored_raw.decode("utf-8"))
            except (ValueError, UnicodeError):
                response = None
            usage = _usage(response)
            if usage is not None:
                input_tokens, output_tokens = usage
                actual = PRICE_PER_MILLION * input_tokens / 1_000_000
                entry.update(input_tokens=input_tokens, output_tokens=output_tokens, actual_cost_usd=_decimal_text(actual))
                if input_tokens > MAX_INPUT_TOKENS:
                    ledger["frozen"] = True
                    entry["provider_context_limit_exceeded"] = True
            error: JevClientError | None = None
            if not 200 <= status < 300:
                entry.update(status="http_error", error_code="http_non_success")
                error = APIRequestFailed(f"API returned HTTP {status}. The response is cached; no retry was made.")
            elif redacted:
                entry.update(status="invalid_response", error_code="credential_echo")
                error = InvalidResponse("API response unexpectedly contained a credential; it was redacted and rejected.")
            elif usage is None:
                entry.update(status="invalid_response", error_code="invalid_usage")
                error = InvalidResponse("API response has no valid token usage. Its maximum-cost reservation remains.")
            elif input_tokens > MAX_INPUT_TOKENS:
                entry.update(status="invalid_response", error_code="context_limit_exceeded")
                error = InvalidResponse("API reported tokens above its standard context limit; the budget is frozen for review.")
            else:
                try:
                    self._validate_response(response, payload)
                except InvalidResponse as exc:
                    entry.update(status="invalid_response", error_code="invalid_answer")
                    error = exc
                else:
                    entry["status"] = "complete"
                    _atomic_json(call_dir / "response.json", response)
            _atomic_json(call_dir / "record.json", entry)
            _atomic_json(self.ledger_path, ledger)
            if error is not None:
                raise error
            return response

    @staticmethod
    def _validate_response(response: Any, payload: Mapping[str, Any]) -> None:
        if _usage(response) is None:
            raise InvalidResponse("Response does not contain valid input and output token usage.")
        if not isinstance(response.get("answers"), dict) or set(response["answers"]) != set(payload["questions"]):
            raise InvalidResponse("Response answer names do not match the requested questions.")
        for name, question in payload["questions"].items():
            parse_choice(response, name, question["criteria"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--budget-usd", default="5")
    parser.add_argument("--payload", type=Path, help="Canonical request JSON; sends one paid request if not cached.")
    parser.add_argument("--request-key")
    parser.add_argument("--context", type=Path, help="Optional JSON provenance object.")
    args = parser.parse_args(argv)
    try:
        client = BudgetedJevClient(args.cache_dir, args.budget_usd)
        if args.payload:
            payload = _read_json(args.payload)
            context = _read_json(args.context) if args.context else None
            response = client.run(payload, args.request_key, context)
            print(json.dumps(response, sort_keys=True, allow_nan=False))
        print(json.dumps({"budget": client.budget_summary()}, sort_keys=True))
        return 0
    except JevClientError as exc:
        print(json.dumps({"error_type": type(exc).__name__, "message": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
