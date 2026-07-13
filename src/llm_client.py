# llm_client.py
# Wrapper around the Groq API for LLM-based clause extraction and summarization.
# Handles rate limiting, retries, key rotation, and text preprocessing.

import os
import re
import time
import logging
import textwrap
from typing import Optional, List

from groq import Groq, RateLimitError, APIStatusError

logger = logging.getLogger(__name__)

DEFAULT_MODEL  = "llama-3.1-8b-instant"
FALLBACK_MODEL = "llama-3.3-70b-versatile"

# keeping tokens low to stay within free tier daily limits
MAX_CONTEXT_TOKENS = 3000
APPROX_CHARS_PER_TOKEN = 4
MAX_CHARS = MAX_CONTEXT_TOKENS * APPROX_CHARS_PER_TOKEN  # ~12000 chars

RETRY_MAX_CHARS = MAX_CHARS // 2  # used when loop detection kicks in

# don't blindly wait 16 minutes when rate limited - cap it
MAX_RATE_LIMIT_WAIT = 120.0


# --- API key rotation ---

def _load_all_keys() -> List[str]:
    """
    Load all Groq API keys from environment.
    Checks GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3 etc.
    """
    keys = []
    k = os.getenv("GROQ_API_KEY", "")
    if k and not k.startswith("YOUR_"):
        keys.append(k)
    for i in range(2, 10):
        k = os.getenv(f"GROQ_API_KEY_{i}", "")
        if k and not k.startswith("YOUR_"):
            keys.append(k)
    return keys


class _KeyRotator:
    """Manages a pool of API keys and rotates when one gets rate limited."""

    def __init__(self):
        self._keys: List[str] = []
        self._idx: int = 0
        self._exhausted: set = set()

    def _refresh(self):
        self._keys = _load_all_keys()
        if not self._keys:
            raise EnvironmentError(
                "No valid GROQ_API_KEY found. "
                "Set GROQ_API_KEY (and optionally GROQ_API_KEY_2) in your .env file."
            )

    def current_client(self) -> Groq:
        if not self._keys:
            self._refresh()
        return Groq(api_key=self._keys[self._idx % len(self._keys)])

    def rotate(self) -> bool:
        """Switch to next available key. Returns False if all keys are exhausted."""
        if not self._keys:
            self._refresh()

        self._exhausted.add(self._idx % len(self._keys))
        for offset in range(1, len(self._keys) + 1):
            next_idx = (self._idx + offset) % len(self._keys)
            if next_idx not in self._exhausted:
                old = (self._idx % len(self._keys)) + 1
                self._idx = next_idx
                new = (self._idx % len(self._keys)) + 1
                logger.warning(f"API key #{old} rate-limited. Switching to key #{new} of {len(self._keys)}.")
                return True

        logger.error(f"All {len(self._keys)} API key(s) are rate-limited.")
        return False

    def reset_exhausted(self):
        self._exhausted.clear()


# singleton - shared across all calls
_rotator = _KeyRotator()


def get_client() -> Groq:
    return _rotator.current_client()


# --- Text preprocessing ---

def _deduplicate_lines(text: str, max_repeats: int = 3) -> str:
    """Remove lines that appear too many times - helps avoid Groq's loop detection."""
    lines = text.splitlines()
    seen_counts: dict = {}
    result = []
    prev = None
    consecutive = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            result.append(line)
            prev = None
            consecutive = 0
            continue

        if stripped == prev:
            consecutive += 1
        else:
            consecutive = 1
            prev = stripped

        if consecutive > max_repeats:
            continue

        seen_counts[stripped] = seen_counts.get(stripped, 0) + 1
        if seen_counts[stripped] > max_repeats * 2:
            continue

        result.append(line)

    return "\n".join(result)


def _remove_noise(text: str) -> str:
    """Strip signature lines, page numbers and other stuff that causes loop detection."""
    text = re.sub(r"[_\-]{5,}", "", text)
    text = re.sub(r"^\s*[-–]?\s*\d{1,3}\s*[-–]?\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_contract_text(text: str, max_chars: int = MAX_CHARS) -> str:
    """Full cleaning pipeline - noise removal, dedup, truncation."""
    text = _remove_noise(text)
    text = _deduplicate_lines(text)
    return truncate_to_token_limit(text, max_chars)


def truncate_to_token_limit(text: str, max_chars: int = MAX_CHARS) -> str:
    """Truncate text to fit within our token budget. Tries to cut at paragraph boundaries."""
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    last_para = truncated.rfind("\n\n")
    if last_para > max_chars * 0.7:
        truncated = truncated[:last_para]

    return truncated + "\n\n[... TEXT TRUNCATED FOR LENGTH LIMITS ...]"


# --- Retry-after parser ---

def _parse_retry_after(error_str: str, default: float = 5.0) -> float:
    """Parse wait time from Groq error messages like 'Please try again in 7m1.9s'."""
    m = re.search(r"try again in\s+(?:(\d+)m)?(?:([\d.]+)s)?", error_str, re.IGNORECASE)
    if m:
        minutes = float(m.group(1) or 0)
        seconds = float(m.group(2) or 0)
        return max(minutes * 60 + seconds + 2, default)  # +2s buffer
    return default


# --- Core LLM call with retry ---

def call_llm(
    prompt: str,
    system_prompt: str = "You are an expert legal contract analyst. Be concise and direct.",
    model: str = DEFAULT_MODEL,
    max_tokens: int = 512,
    temperature: float = 0.1,
    max_retries: int = 5,
    initial_wait: float = 2.0,
) -> str:
    """
    Call the Groq LLM with automatic retry on rate limits and server errors.
    Also handles Groq's loop-detection errors by shortening the prompt.
    """
    client = get_client()
    wait = initial_wait
    current_prompt = prompt

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": current_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                frequency_penalty=0.3,
                presence_penalty=0.1,
            )
            return response.choices[0].message.content.strip()

        except RateLimitError as e:
            if attempt == max_retries:
                raise
            # try switching to another API key first
            if _rotator.rotate():
                client = get_client()
                logger.info(f"  Retrying immediately with new API key (attempt {attempt}/{max_retries})...")
                continue
            # all keys exhausted - wait before retrying
            raw_wait = _parse_retry_after(str(e), default=wait)
            wait = min(raw_wait, MAX_RATE_LIMIT_WAIT)
            logger.warning(
                f"All keys rate-limited (attempt {attempt}/{max_retries}). "
                f"Waiting {wait:.0f}s (API requested {raw_wait:.0f}s)..."
            )
            time.sleep(wait)
            wait = min(wait * 1.5, MAX_RATE_LIMIT_WAIT)

        except APIStatusError as e:
            err_body = str(e).lower()

            if "model_decommissioned" in err_body or "decommissioned" in err_body:
                raise RuntimeError(
                    f"Model '{model}' has been decommissioned by Groq. "
                    f"Use --model llama-3.1-8b-instant or llama-3.3-70b-versatile."
                ) from e

            if "looping content" in err_body or "loop detection" in err_body:
                if attempt == max_retries:
                    logger.error(f"Loop detection persists after {max_retries} retries.")
                    raise
                logger.warning(
                    f"Loop-detection triggered (attempt {attempt}/{max_retries}). "
                    "Reducing and cleaning contract text..."
                )
                current_prompt = _shorten_prompt_for_retry(current_prompt, attempt)
                time.sleep(1.0)
                continue

            if e.status_code in (503, 529) and attempt < max_retries:
                logger.warning(f"API overloaded ({e.status_code}), waiting {wait:.1f}s...")
                time.sleep(wait)
                wait = min(wait * 2, 120)
            else:
                raise

    return ""


def _shorten_prompt_for_retry(prompt: str, attempt: int) -> str:
    """Cut down the contract text on loop-detection retries. Each attempt halves it."""
    marker = "CONTRACT TEXT:"
    idx = prompt.find(marker)
    if idx == -1:
        limit = len(prompt) // (attempt + 1)
        return prompt[:limit] + "\n\n[TEXT SHORTENED DUE TO RETRY]"

    header = prompt[:idx + len(marker)]
    body = prompt[idx + len(marker):]

    limit = max(2000, len(body) // (2 ** attempt))
    short_body = _deduplicate_lines(body)
    short_body = truncate_to_token_limit(short_body.strip(), limit)

    footer_marker = "[... TEXT TRUNCATED"
    footer_idx = short_body.find(footer_marker)
    if footer_idx > 0:
        short_body = short_body[:footer_idx]

    original_parts = prompt.split(marker, 1)
    new_prompt = (
        original_parts[0]
        + marker + "\n"
        + short_body
        + "\n\nIMPORTANT: Respond ONLY with the three labeled lines below. "
        "Do not repeat any text from the contract.\n"
    )
    if len(original_parts) > 1:
        orig_body = original_parts[1]
        respond_idx = orig_body.rfind("Respond in this EXACT format")
        if respond_idx == -1:
            respond_idx = orig_body.rfind("Write the summary")
        if respond_idx > 0:
            new_prompt += orig_body[respond_idx:]

    return new_prompt


# --- High-level functions used by extractor.py ---

def extract_clauses(contract_text: str, model: str = DEFAULT_MODEL, few_shot: bool = True) -> dict:
    """Extract termination, confidentiality, and liability clauses from a contract."""
    safe_text = clean_contract_text(contract_text)

    few_shot_examples = ""
    if few_shot:
        few_shot_examples = textwrap.dedent("""
        ### EXAMPLES (for format reference only) ###

        EXAMPLE 1
        TERMINATION: Either party may terminate upon 30 days written notice; \
the Company may terminate immediately upon material breach.

        EXAMPLE 2
        CONFIDENTIALITY: Each party must keep the other's proprietary information \
confidential and must not disclose it to third parties.

        EXAMPLE 3
        LIABILITY: Neither party is liable for indirect, incidental, or \
consequential damages arising from this agreement.

        ### END EXAMPLES ###
        """)

    prompt = textwrap.dedent(f"""
    {few_shot_examples}

    Read the legal contract below. Extract ONE concise sentence (max 40 words each)
    for each of the three clause types. If a clause is absent, write:
    "Not specified in this contract."

    CONTRACT TEXT:
    {safe_text}

    Respond in this EXACT format (three lines only, no extra text):
    TERMINATION: <one sentence>
    CONFIDENTIALITY: <one sentence>
    LIABILITY: <one sentence>
    """).strip()

    raw = call_llm(prompt, model=model, max_tokens=300, temperature=0.1)
    return _parse_clause_response(raw)


def _parse_clause_response(raw: str) -> dict:
    result = {
        "termination_clause":     "Not specified in this contract.",
        "confidentiality_clause": "Not specified in this contract.",
        "liability_clause":       "Not specified in this contract.",
    }

    for line in raw.splitlines():
        line = line.strip()
        upper = line.upper()
        if upper.startswith("TERMINATION:"):
            val = line.split(":", 1)[1].strip()
            if val:
                result["termination_clause"] = val
        elif upper.startswith("CONFIDENTIALITY:"):
            val = line.split(":", 1)[1].strip()
            if val:
                result["confidentiality_clause"] = val
        elif upper.startswith("LIABILITY:"):
            val = line.split(":", 1)[1].strip()
            if val:
                result["liability_clause"] = val

    return result


def summarize_contract(contract_text: str, model: str = DEFAULT_MODEL) -> str:
    """Generate a short summary of the contract - purpose, obligations, risks."""
    safe_text = clean_contract_text(contract_text)

    prompt = textwrap.dedent(f"""
    Read the legal contract below. Write a single paragraph summary of \
100-150 words covering:
    1. The type of agreement and parties involved.
    2. Key obligations of each party.
    3. Notable risks, penalties, or important conditions.

    CONTRACT TEXT:
    {safe_text}

    Write ONLY the summary paragraph. No headings, no bullet points, \
no repetition of contract text.
    """).strip()

    return call_llm(prompt, model=model, max_tokens=300, temperature=0.2)


def extract_and_summarize(contract_text: str, model: str = DEFAULT_MODEL, few_shot: bool = True) -> dict:
    """
    Combined extraction + summarization in a single API call.
    More efficient than calling extract_clauses() and summarize_contract() separately.
    Returns: termination_clause, confidentiality_clause, liability_clause, summary
    """
    safe_text = clean_contract_text(contract_text)

    few_shot_examples = ""
    if few_shot:
        few_shot_examples = textwrap.dedent("""
        ### FORMAT EXAMPLES ###
        TERMINATION: Either party may terminate upon 30 days written notice.
        CONFIDENTIALITY: Each party must keep proprietary information confidential.
        LIABILITY: Neither party is liable for indirect or consequential damages.
        SUMMARY: This is a software licensing agreement between Acme Corp and Beta Ltd...
        ### END EXAMPLES ###
        """)

    prompt = textwrap.dedent(f"""
    {few_shot_examples}

    Read the legal contract below. In ONE response, provide ALL of the following:
    1. One concise sentence (max 40 words) for each clause type.
    2. A 80-120 word summary covering: agreement type, key obligations, notable risks.
    If a clause is absent, write: "Not specified in this contract."

    CONTRACT TEXT:
    {safe_text}

    Respond in this EXACT format (4 lines only, no extra text):
    TERMINATION: <one sentence>
    CONFIDENTIALITY: <one sentence>
    LIABILITY: <one sentence>
    SUMMARY: <one paragraph>
    """).strip()

    raw = call_llm(prompt, model=model, max_tokens=500, temperature=0.1)
    return _parse_combined_response(raw)


def _parse_combined_response(raw: str) -> dict:
    result = {
        "termination_clause":     "Not specified in this contract.",
        "confidentiality_clause": "Not specified in this contract.",
        "liability_clause":       "Not specified in this contract.",
        "summary":                "",
    }

    summary_parts = []
    in_summary = False

    for line in raw.splitlines():
        stripped = line.strip()
        upper = stripped.upper()

        if upper.startswith("TERMINATION:"):
            val = stripped.split(":", 1)[1].strip()
            if val:
                result["termination_clause"] = val
            in_summary = False
        elif upper.startswith("CONFIDENTIALITY:"):
            val = stripped.split(":", 1)[1].strip()
            if val:
                result["confidentiality_clause"] = val
            in_summary = False
        elif upper.startswith("LIABILITY:"):
            val = stripped.split(":", 1)[1].strip()
            if val:
                result["liability_clause"] = val
            in_summary = False
        elif upper.startswith("SUMMARY:"):
            val = stripped.split(":", 1)[1].strip()
            if val:
                summary_parts.append(val)
            in_summary = True
        elif in_summary and stripped:
            summary_parts.append(stripped)

    result["summary"] = " ".join(summary_parts)
    return result
