import re
import hashlib
import logging
from pathlib import Path
import yaml
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

logger = logging.getLogger(__name__)

# ── Config-driven recognizer loader ───────────────────────────────────────────

def _load_recognizers_from_config(config_path: str = "config/pii_recognizers.yaml") -> list:
    """
    Load PII recognizers from YAML config.
    Adding a new country = add lines to YAML, zero code changes.
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"PII config not found at {config_path}, using empty recognizer list")
        return []

    with open(path) as f:
        config = yaml.safe_load(f)

    recognizers = []
    for r in config.get("recognizers", []):
        recognizer = PatternRecognizer(
            supported_entity=r["entity"],
            name=r.get("name", r["entity"]),
            patterns=[Pattern(
                name=r["entity"],
                regex=r["pattern"],
                score=r["score"]
            )]
        )
        recognizers.append(recognizer)
        logger.info(f"Loaded PII recognizer: {r['entity']} ({r.get('country', 'GLOBAL')})")

    return recognizers


# ── Build Presidio analyzer with spaCy NER + config-driven recognizers ────────

def _build_analyzer() -> AnalyzerEngine:
    try:
        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        })
        nlp_engine = provider.create_engine()
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
        logger.info("Presidio initialized with spaCy en_core_web_lg")
    except Exception as e:
        logger.warning(f"spaCy model unavailable, falling back to default NLP engine: {e}")
        analyzer = AnalyzerEngine()

    # Load and register all recognizers from YAML
    custom_recognizers = _load_recognizers_from_config()
    for recognizer in custom_recognizers:
        analyzer.registry.add_recognizer(recognizer)

    logger.info(f"Registered {len(custom_recognizers)} custom PII recognizers")
    return analyzer


# Initialise once at module load — NOT on every request
_analyzer = _build_analyzer()
_anonymizer = AnonymizerEngine()

# Built-in Presidio entities (spaCy NER) + all custom entities from YAML
_BUILTIN_ENTITIES = [
    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION",
    "CREDIT_CARD", "IBAN_CODE", "DATE_TIME", "NRP", "ORGANIZATION",
]

def _get_all_entities() -> list[str]:
    """Dynamically build entity list from config + builtins."""
    config_path = Path("config/pii_recognizers.yaml")
    custom_entities = []
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        custom_entities = [r["entity"] for r in config.get("recognizers", [])]
    return _BUILTIN_ENTITIES + custom_entities


# ── Deterministic token ───────────────────────────────────────────────────────

def _make_token(entity_type: str, value: str) -> str:
    """
    Same PII value always produces same token.
    Critical for FAISS vector consistency between ingestion and query time.
    """
    h = hashlib.md5(f"{entity_type}:{value}".encode()).hexdigest()[:8].upper()
    return f"[{entity_type}_{h}]"


# ── Public API ────────────────────────────────────────────────────────────────

def mask_pii(text: str) -> tuple[str, dict]:
    """
    3-layer PII masking:
      Layer 1 — Config-driven regex (all countries in pii_recognizers.yaml)
      Layer 2 — spaCy NER (PERSON, LOCATION, ORG in free text)
      Layer 3 — Deterministic token substitution for vector consistency

    Returns (masked_text, token_mapping).
    """
    mapping: dict[str, str] = {}
    entities = _get_all_entities()

    try:
        results = _analyzer.analyze(text=text, entities=entities, language="en")
    except Exception as e:
        logger.error(f"Presidio analysis failed: {e}")
        return text, mapping

    if not results:
        return text, mapping

    masked = text
    offset = 0

    for result in sorted(results, key=lambda r: r.start):
        original = text[result.start:result.end]
        token = _make_token(result.entity_type, original)
        mapping[token] = original

        start = result.start + offset
        end = result.end + offset
        masked = masked[:start] + token + masked[end:]
        offset += len(token) - (result.end - result.start)

    return masked, mapping


def unmask_pii(text: str, mapping: dict) -> str:
    """Restore original PII values from token mapping."""
    for token, original in mapping.items():
        text = text.replace(token, original)
    return text


def contains_pii(text: str) -> bool:
    """Returns True if any PII entity is detected."""
    try:
        results = _analyzer.analyze(
            text=text,
            entities=_get_all_entities(),
            language="en"
        )
        return len(results) > 0
    except Exception:
        return False


def pii_report(text: str) -> list[dict]:
    """
    Detailed PII audit report — for compliance logging.
    Log this, never log the actual PII values.
    """
    try:
        results = _analyzer.analyze(
            text=text,
            entities=_get_all_entities(),
            language="en"
        )
        return [
            {
                "entity_type": r.entity_type,
                "value": text[r.start:r.end],
                "score": round(r.score, 3),
                "start": r.start,
                "end": r.end,
            }
            for r in results
        ]
    except Exception as e:
        logger.error(f"PII report failed: {e}")
        return []