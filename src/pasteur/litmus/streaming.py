"""SSE streaming with blinded timing for LITMUS (Algorithm 1).

Implements the blinded token streaming protocol from the paper:
all entity sources (MARE, Amalgam, Real) are streamed at an identical
rate derived from Amalgam's generation profile, preventing participants
from identifying the source by delivery timing.
"""

import json
import math
import random
import time
from dataclasses import dataclass

from .data import fixup_partial_json


@dataclass
class StreamingParams:
    """Parameters for blinded token streaming (Algorithm 1)."""

    tps_0: float = 15.0  # Initial tokens per second (from Amalgam profile)
    gamma: float = 0.03  # Decay factor for throughput reduction
    t_wait: float = 2.0  # Universal wait time before first token (seconds)
    sigma_start: float = 0.5  # Startup jitter range (uniform)
    sigma: float = 0.02  # Per-token jitter std dev (normal)

    @classmethod
    def default(cls) -> "StreamingParams":
        return cls()


def compute_delivery_schedule(
    n_tokens: int, params: StreamingParams
) -> list[float]:
    """Compute absolute delivery times for n tokens using Algorithm 1.

    Returns a list of absolute times (in seconds) for each token,
    starting from t=0 (request time).
    """
    # Phase 1: Universal wait with startup jitter
    epsilon_start = random.uniform(0, params.sigma_start)
    t_wait = params.t_wait + epsilon_start

    # Phase 2: Compute delivery schedule
    schedule = []
    cumulative = 0.0

    for i in range(1, n_tokens + 1):
        # Inter-token interval with logarithmic decay
        delta = (1.0 / params.tps_0) * (1.0 + params.gamma * math.log(1 + i))
        cumulative += delta

        # Per-token jitter
        epsilon_i = random.gauss(0, params.sigma)

        # Absolute delivery time
        t_i = t_wait + cumulative + epsilon_i
        schedule.append(max(t_i, schedule[-1] + 0.001 if schedule else t_wait))

    return schedule


def tokenize_json(json_str: str, chunk_size: int = 4) -> list[str]:
    """Split a JSON string into display tokens.

    Uses character-level chunking that respects word boundaries where
    possible. Each token is a small string fragment.

    Args:
        json_str: The complete JSON string to tokenize.
        chunk_size: Approximate characters per token.

    Returns:
        List of string tokens that concatenate to the original.
    """
    tokens = []
    i = 0
    while i < len(json_str):
        end = min(i + chunk_size, len(json_str))

        # Try to break at a word boundary (space, comma, colon)
        if end < len(json_str):
            for break_char in (" ", ",", ":", "\n"):
                pos = json_str.rfind(break_char, i, end + 1)
                if pos > i:
                    end = pos + 1
                    break

        tokens.append(json_str[i:end])
        i = end

    return tokens


def generate_sse_stream(
    entity: dict,
    entity_id: str,
    source: str,
    params: StreamingParams | None = None,
    blind: bool = True,
):
    """Generator that yields SSE events for streaming an entity.

    Implements Algorithm 1: blinded token delivery protocol.

    Yields SSE-formatted strings: "data: {...}\n\n"

    Events:
    - type=start: Entity metadata (entity_id)
    - type=token: Partial token with accumulated data for progressive display
    - type=done: Entity fully delivered, includes source for rating

    Args:
        entity: The entity dict to stream.
        entity_id: Unique ID for this entity.
        source: Source label (e.g., "mare_2026-01-23", "real"). Hidden from client until done.
        params: Streaming parameters. Uses defaults if None.
        blind: If True, apply blinded timing. If False, send all at once.
    """
    if params is None:
        params = StreamingParams.default()

    entity_json = json.dumps(entity, indent=2)

    # Send start event immediately
    yield _sse_event({"type": "start", "entity_id": entity_id})

    if not blind:
        # Non-blinded: send complete entity at once
        yield _sse_event({
            "type": "token",
            "fragment": entity_json,
            "accumulated": entity_json,
            "pretty": entity_json,
            "progress": 1.0,
        })
        yield _sse_event({
            "type": "done",
            "entity_id": entity_id,
            "source": source,
        })
        return

    # Blinded streaming: tokenize and apply timing schedule
    tokens = tokenize_json(entity_json)
    schedule = compute_delivery_schedule(len(tokens), params)

    start_time = time.monotonic()
    accumulated = ""

    for i, (token, target_time) in enumerate(zip(tokens, schedule)):
        # Wait until scheduled delivery time
        now = time.monotonic() - start_time
        if target_time > now:
            time.sleep(target_time - now)

        accumulated += token
        progress = (i + 1) / len(tokens)

        # Try to produce a pretty-printed version of the partial JSON
        pretty = fixup_partial_json(accumulated)

        yield _sse_event({
            "type": "token",
            "fragment": token,
            "accumulated": accumulated,
            "pretty": pretty,
            "progress": progress,
        })

    # Send completion event
    yield _sse_event({
        "type": "done",
        "entity_id": entity_id,
        "source": source,
    })


def _sse_event(data: dict) -> str:
    """Format a dict as an SSE event string."""
    return f"data: {json.dumps(data)}\n\n"
