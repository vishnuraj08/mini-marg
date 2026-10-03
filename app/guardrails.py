import re
import logging
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


# ── Enums ─────────────────────────────────────────────────────────────────────

class GuardrailAction(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    REDACT = "redact"
    WARN = "warn"


class GuardrailLayer(str, Enum):
    INPUT = "input"
    OUTPUT = "output"


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    action: GuardrailAction
    layer: GuardrailLayer
    triggered_rule: str | None = None
    reason: str | None = None
    sanitized_text: str | None = None


# ── Rule definitions (loaded from config) ────────────────────────────────────

def _load_rules(config_path: str = "config/guardrails.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"Guardrails config not found at {config_path}, using defaults")
        return _default_rules()
    with open(path) as f:
        return yaml.safe_load(f)


def _default_rules() -> dict:
    return {
        "input": {
            "blocked_patterns": [],
            "max_length": 2000,
            "blocked_topics": [],
        },
        "output": {
            "blocked_patterns": [],
            "max_length": 5000,
        }
    }


# ── Individual guardrail checks ───────────────────────────────────────────────

def _check_length(text: str, max_length: int, layer: GuardrailLayer) -> GuardrailResult | None:
    if len(text) > max_length:
        return GuardrailResult(
            action=GuardrailAction.BLOCK,
            layer=layer,
            triggered_rule="max_length",
            reason=f"Text length {len(text)} exceeds limit of {max_length}"
        )
    return None


def _check_blocked_patterns(text: str, patterns: list, layer: GuardrailLayer) -> GuardrailResult | None:
    for rule in patterns:
        if re.search(rule["pattern"], text, re.IGNORECASE):
            action = GuardrailAction(rule.get("action", "block"))
            return GuardrailResult(
                action=action,
                layer=layer,
                triggered_rule=rule["name"],
                reason=rule.get("reason", f"Matched blocked pattern: {rule['name']}")
            )
    return None


def _check_blocked_topics(text: str, topics: list, layer: GuardrailLayer) -> GuardrailResult | None:
    text_lower = text.lower()
    for topic in topics:
        keywords = topic.get("keywords", [])
        if any(kw.lower() in text_lower for kw in keywords):
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                layer=layer,
                triggered_rule=topic["name"],
                reason=topic.get("reason", f"Topic not permitted: {topic['name']}")
            )
    return None


def _check_prompt_injection(text: str, layer: GuardrailLayer) -> GuardrailResult | None:
    """
    Detect prompt injection attempts — user trying to override system instructions.
    Common in BFSI where attackers try to extract system prompts or bypass rules.
    """
    injection_patterns = [
        r"ignore (previous|all|above|prior) instructions",
        r"forget (everything|all|your instructions)",
        r"you are now",
        r"act as (a|an|if)",
        r"disregard (your|all|previous)",
        r"system prompt",
        r"reveal (your|the) (prompt|instructions|system)",
        r"bypass (safety|guardrails|filters|restrictions)",
        r"jailbreak",
        r"DAN mode",
    ]
    for pattern in injection_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return GuardrailResult(
                action=GuardrailAction.BLOCK,
                layer=layer,
                triggered_rule="prompt_injection",
                reason="Potential prompt injection attempt detected"
            )
    return None


def _check_hallucination_markers(text: str) -> GuardrailResult | None:
    """
    Output-only check: detect LLM hallucination signals.
    BFSI critical — wrong interest rates or regulatory info = compliance risk.
    """
    hallucination_signals = [
        r"as of my (knowledge|training) cutoff",
        r"i (don't|do not) have (access|information)",
        r"i('m| am) not sure (but|however)",
        r"this (may|might|could) (be|have) changed",
        r"please (verify|confirm|check) (this|with)",
        r"i (cannot|can't) guarantee",
    ]
    for pattern in hallucination_signals:
        if re.search(pattern, text, re.IGNORECASE):
            return GuardrailResult(
                action=GuardrailAction.WARN,
                layer=GuardrailLayer.OUTPUT,
                triggered_rule="hallucination_signal",
                reason="LLM output contains uncertainty markers — human review recommended"
            )
    return None


def _check_sensitive_financial_claims(text: str) -> GuardrailResult | None:
    """
    Output-only: block definitive financial/legal advice without disclaimers.
    Regulatory requirement in BFSI — LLM cannot give guaranteed financial advice.
    """
    advice_patterns = [
        r"you (must|should definitely|are guaranteed to)",
        r"guaranteed (return|profit|income)",
        r"100% (safe|secure|guaranteed)",
        r"(definitely|certainly) (invest|buy|sell)",
    ]
    for pattern in advice_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return GuardrailResult(
                action=GuardrailAction.REDACT,
                layer=GuardrailLayer.OUTPUT,
                triggered_rule="definitive_financial_advice",
                reason="Output contains unqualified financial advice — adding disclaimer"
            )
    return None


# ── Main public functions ─────────────────────────────────────────────────────

_rules = _load_rules()


def check_input(text: str) -> GuardrailResult:
    """
    Run all input guardrails before query reaches RAG pipeline.
    Order: injection → length → blocked patterns → blocked topics
    """
    input_rules = _rules.get("input", {})

    # 1. Prompt injection (highest priority — always check)
    result = _check_prompt_injection(text, GuardrailLayer.INPUT)
    if result:
        logger.warning(f"INPUT BLOCKED [{result.triggered_rule}]: {result.reason}")
        return result

    # 2. Length check
    max_len = input_rules.get("max_length", 2000)
    result = _check_length(text, max_len, GuardrailLayer.INPUT)
    if result:
        logger.warning(f"INPUT BLOCKED [{result.triggered_rule}]: {result.reason}")
        return result

    # 3. Blocked patterns from config
    blocked_patterns = input_rules.get("blocked_patterns", [])
    result = _check_blocked_patterns(text, blocked_patterns, GuardrailLayer.INPUT)
    if result:
        logger.warning(f"INPUT BLOCKED [{result.triggered_rule}]: {result.reason}")
        return result

    # 4. Blocked topics from config
    blocked_topics = input_rules.get("blocked_topics", [])
    result = _check_blocked_topics(text, blocked_topics, GuardrailLayer.INPUT)
    if result:
        logger.warning(f"INPUT BLOCKED [{result.triggered_rule}]: {result.reason}")
        return result

    return GuardrailResult(action=GuardrailAction.ALLOW, layer=GuardrailLayer.INPUT)


def check_output(text: str) -> GuardrailResult:
    """
    Run all output guardrails before response reaches the user.
    Order: length → blocked patterns → hallucination → financial advice
    """
    output_rules = _rules.get("output", {})

    # 1. Length check
    max_len = output_rules.get("max_length", 5000)
    result = _check_length(text, max_len, GuardrailLayer.OUTPUT)
    if result:
        logger.warning(f"OUTPUT BLOCKED [{result.triggered_rule}]: {result.reason}")
        return result

    # 2. Blocked patterns from config
    blocked_patterns = output_rules.get("blocked_patterns", [])
    result = _check_blocked_patterns(text, blocked_patterns, GuardrailLayer.OUTPUT)
    if result:
        logger.warning(f"OUTPUT BLOCKED [{result.triggered_rule}]: {result.reason}")
        return result

    # 3. Hallucination signals (WARN — don't block, but flag)
    result = _check_hallucination_markers(text)
    if result:
        logger.warning(f"OUTPUT WARNED [{result.triggered_rule}]: {result.reason}")
        return result

    # 4. Definitive financial advice (REDACT — append disclaimer)
    result = _check_sensitive_financial_claims(text)
    if result:
        result.sanitized_text = (
            text + "\n\n⚠️ Disclaimer: This information is for reference only "
            "and does not constitute financial advice. Please consult a qualified advisor."
        )
        logger.warning(f"OUTPUT REDACTED [{result.triggered_rule}]: {result.reason}")
        return result

    return GuardrailResult(action=GuardrailAction.ALLOW, layer=GuardrailLayer.OUTPUT)


def apply_output(original_text: str, result: GuardrailResult) -> str:
    """
    Apply the guardrail result to the output text.
    Call this after check_output() to get the final text to send to user.
    """
    if result.action == GuardrailAction.BLOCK:
        return f"I'm unable to respond to this request. Reason: {result.reason}"
    if result.action == GuardrailAction.REDACT and result.sanitized_text:
        return result.sanitized_text
    if result.action == GuardrailAction.WARN:
        return original_text + "\n\n⚠️ Note: This response may require human verification."
    return original_text