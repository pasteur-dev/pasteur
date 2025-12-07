import threading
import time
import logging
from typing import Any, Literal, Mapping, Type, TypedDict
from pasteur.utils import LazyDataset
from pasteur.utils.progress import prange
import pandas as pd
import numpy as np

PRINT_FREQ = 0.2
TOP_K = 3
MAX_EXPOSURE = 5
PART_SIZE = 5000
THINK = False
MAX_FAILS = 20

logger = logging.getLogger(__name__)


class CacheTracker:
    def __init__(self, cache_time: float | None = None, cache_len: int | None = None):
        self.lock = threading.Lock()
        self.cache = {}
        self.idx = 0

        self.cache_time = cache_time
        self.cache_len = cache_len
        self.cached_tokens = 0

    def _get_cached_len(self, tokens: list[int], ctime=None):
        curr = self.cache

        for i, t in enumerate(tokens):
            if t not in curr:
                return i
            if ctime is not None and ctime > curr[t].get("_ctime", 0):
                return i
            if self.cache_len and self.idx - self.cache_len > curr[t].get("_cidx", 0):
                return i
            curr = curr[t]

        return len(tokens)

    def get_cached_len(self, tokens: list[int]):
        with self.lock:
            i = self._get_cached_len(
                tokens,
                time.perf_counter() - self.cache_time if self.cache_time else None,
            )
            self.cached_tokens += i
            return i

    def _add_cached_tokens(self, tokens: list[int]):
        curr = self.cache

        for t in tokens:
            if t not in curr:
                curr[t] = {}

            curr["_ctime"] = time.perf_counter()
            curr["_cidx"] = self.idx
            curr = curr[t]

        self.idx += 1

    def add_cached_tokens(self, tokens: list[int]):
        with self.lock:
            self._add_cached_tokens(tokens)


class AmalgamHFParams(TypedDict):
    type: Literal["hf"]
    repo_id: str
    filename: str
    n_ctx: int
    n_gpu_layers: int
    workers: int


class AmalgamORParams(TypedDict):
    type: Literal["or"]
    model: str
    workers: int


def _load_llm_model(params: AmalgamHFParams | AmalgamORParams, output_type) -> Any:
    import outlines
    from outlines import Generator

    match params["type"]:
        case "hf":
            from llama_cpp import Llama
            from outlines.models import LlamaCpp

            llm = LlamaCpp(
                Llama.from_pretrained(
                    repo_id=params["repo_id"],
                    filename=params["filename"],
                    n_ctx=params["n_ctx"],
                    n_gpu_layers=params["n_gpu_layers"],
                    batch_size=1,
                    verbose=False,
                )
            )
        case "or":
            import openai

            # Create the model
            llm = outlines.from_openai(
                openai.OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=get_or_api_key(),
                ),
                params["model"],
            )

    generator_thought = Generator(llm)
    # Generating the
    generator = Generator(llm, output_type)

    return {
        "model_type": params["type"],
        "llm": llm,
        "generator": generator,
        "generator_thought": generator_thought,
    }


def load_llm_model(
    params: AmalgamHFParams | AmalgamORParams,
    output_type,
):
    llms = []

    for _ in range(params.get("workers", 1)):
        llms.append(_load_llm_model(params, output_type))

    return {
        "model_type": params["type"],
        "llms": llms,
    }


def get_or_api_key() -> str:
    from pasteur.kedro import context

    assert context is not None, "Kedro context is not initialized."
    return context._get_config_credentials()["openrouter"]


def _printer(prompt, sample_num, sample_n, q):
    import json

    MAX_LEN = 300
    prompt_reduced = "\n".join(
        line if len(line) <= MAX_LEN else line[:MAX_LEN] + "..."
        for line in prompt.split("\n")
    )

    decoder = json.JSONDecoder()
    thought = ""
    data = ""
    ttft = None
    last_print = time.perf_counter()
    while token := q.get():
        dtype, j = token

        if j is None:
            break
        if ttft is None:
            ttft = time.perf_counter()
        if isinstance(j, str):
            frac = j
        else:
            assert "object" in j and j["object"] == "text_completion"
            frac = j["choices"][0]["text"]  # type: ignore

        if dtype == "thought":
            thought += frac
        else:
            data += frac

        # Flush the queue before printing
        # Printing is slow, so we only want to print the latest
        if not q.empty():
            continue

        curr = time.perf_counter()
        if curr - last_print < PRINT_FREQ:
            continue

        # Try to correct invalid json as much as possible
        # This does not work in dicts, when before the :
        suffix = ""
        for i, d in enumerate(data):
            if d == "{":
                suffix += "}"
            elif d == "[":
                suffix += "]"
            elif d == '"':
                if suffix.endswith('"'):
                    suffix = suffix[:-1]
                else:
                    suffix += '"'
            elif d == "}":
                if suffix.endswith("}"):
                    suffix = suffix[:-1]
            elif d == "]":
                if suffix.endswith("]"):
                    suffix = suffix[:-1]

        suffix = suffix[::-1]  # reverse

        sdata = data.rstrip()
        if sdata.endswith(","):
            full = data[:-1] + suffix
        elif sdata.endswith(":"):
            full = data + " null" + suffix
        else:
            full = data + suffix

        try:
            if full:
                obj, end = decoder.raw_decode(full)
                pretty = json.dumps(obj, indent=2) + "\n"
            else:
                pretty = ""

            thought_str = ""
            if thought:
                thought_str = f"\nThought: {thought}"
            logger.info(
                f":ephemeral:Sampling Entity {sample_num}/{sample_n}. Prompt: {prompt_reduced}{thought_str}\nData:\n{pretty}"
            )
            last_print = curr
        except json.JSONDecodeError:
            pass


def _worker(
    generator,
    generator_thought,
    stop,
    in_q,
    out_q,
    sample_n,
    think,
    print,
):
    import queue

    while not stop.is_set():
        try:
            prompt, sample_num = in_q.get(timeout=0.1)
        except queue.Empty:
            continue

        start = time.perf_counter()
        ttft = None
        ttft_thought = None

        full_prompt = prompt
        pq = queue.Queue()

        if print:
            t = threading.Thread(
                target=_printer, args=(prompt, sample_num, sample_n, pq), daemon=True
            )
            t.start()
        else:
            t = None

        data = []
        try:
            if think:
                full_prompt = prompt + "\\think<think>"
                for j in generator_thought.stream(full_prompt, max_tokens=None, stop="</think>"):  # type: ignore
                    if ttft_thought is None:
                        ttft_thought = time.perf_counter()
                    if stop.is_set():
                        break
                    full_prompt += j
                    pq.put(("thought", j))
                    data.append(("thought", j))

            for j in generator.stream(full_prompt, max_tokens=None):  # type: ignore
                if ttft is None:
                    ttft = time.perf_counter()
                if stop.is_set():
                    break
                pq.put(("data", j))
                data.append(("data", j))
        except Exception:
            import traceback

            logger.error(f"Error in thought worker:\n{traceback.format_exc()}")
        finally:
            end = time.perf_counter()
            out_q.put((start, ttft_thought, ttft, end, data))

            pq.put(None)
            if t is not None:
                t.join()


def _process_name(name):
    if "_count" not in name:
        return name

    return "NUMBER OF " + name.replace("_count", "").upper()


def _process_val(val):
    if isinstance(val, dict):
        return {k: _process_val(v) for k, v in val.items()}

    if not isinstance(val, str):
        return str(val)

    if not val.startswith("[") or not val.endswith(")"):
        return val

    # Convert bounds like [2, 3) to integer 2 to be simpler to parse
    v1, v2 = map(lambda s: float(s.strip()), val[1:-1].split(","))
    if v1.is_integer() and v2.is_integer() and v2 == v1 + 1:
        return int(v1)

    return val


def _sample(
    gen,
    prompt: str,
    counts,
    meta: Any,
    syn: pd.DataFrame,
    ref: pd.DataFrame,
    data: dict[str, LazyDataset],
    _ctx,
):
    import json
    import queue
    import random
    import threading
    import time

    import numpy as np

    from IPython.display import HTML, display

    from pasteur.extras.encoders import create_table_mapping, process_entity

    out = []

    decoder = json.JSONDecoder()
    topk_idx = []
    topk_scores = []

    max_per_val = 2**16 // len(counts)
    norm_scores = {
        k: np.where(
            v > 0, max_per_val * v[v != 0].min() / np.where(v > 0, v, 1), 0
        ).astype(np.uint16)
        for k, v in counts.items()
        if not k.endswith("_common") and not k.endswith("_cmn")
    }

    id_map = []
    part_map = {}

    for i, (k, v) in enumerate(data["ids"].items()):
        part_map[i] = k
        uv = v()
        id_map.append(uv.rename(columns={next(iter(uv.columns)): "id"}).assign(part=i))

    id_map = pd.concat(id_map).set_index("id")

    for p in range(0, len(ref) // PART_SIZE + 1):
        start = p * PART_SIZE
        end = min((p + 1) * PART_SIZE, len(ref))

        scores = np.zeros((end - start, len(syn)), np.uint16)

        for k, v in norm_scores.items():
            mult = 5 if "_total_count" in k else 1

            vals = v[ref.iloc[start:end][k].values[:, None]]
            eqs = (
                ref.iloc[start:end][k].values[:, None] == syn.loc[:, k].values[None, :]
            )
            scores += mult * (vals * eqs).astype(np.uint16)

        idx = scores.argsort(axis=0)[-TOP_K:, :]
        scores_topk = np.take_along_axis(scores, idx, axis=0)
        idx_index = np.take_along_axis(
            np.array(ref.index), start + idx.flatten()
        ).reshape(idx.shape)

        topk_idx.append(idx_index)
        topk_scores.append(scores_topk)

    full_idx = np.concat(topk_idx)
    full_scores = np.concat(topk_scores)

    lookups = np.take_along_axis(
        full_idx, full_scores.argsort(axis=0)[-TOP_K:, :], axis=0
    )

    t = None
    stop = _ctx["stop"]

    jdata = data["data"]
    n_samples = syn.shape[0]

    fails = 0

    in_q = queue.Queue()
    out_q = queue.Queue()

    for i, llm in enumerate(gen["llms"]):
        t = threading.Thread(
            target=_worker,
            args=(
                llm["generator"],
                llm["generator_thought"],
                stop,
                in_q,
                out_q,
                n_samples,
                THINK,
                i == 0,
            ),
            daemon=True,
        )
        t.start()
        _ctx["t"].append(t)

    for i in range(n_samples):
        samples = []
        for k in lookups[:, i].tolist():
            arr = jdata[part_map[int(id_map.loc[k].iloc[0])]]()
            val = arr.loc[k, next(iter(arr.columns))]
            samples.append(str(val))

        samples = "\n".join(samples)

        base_data = process_entity(
            "table",
            i,
            create_table_mapping("table", {}, {"table": meta["flat"]["meta"]}, {}),
            {"table": syn},
            {},
            {},
        )
        sample_num = i + 1
        seed = "\n".join(
            f"{_process_name(k)}: {_process_val(v)}" for k, v in base_data.items()
        )

        fprompt = (
            prompt.replace("<seed>", seed)
            .replace("<samples>", samples)
            .replace("<samples_n>", str(TOP_K))
        )

        in_q.put((fprompt, sample_num))
    
    for i in prange(n_samples):
        start, ttft_thought, ttft, end, chunks = out_q.get()

        data = ""
        for d in chunks:
            dtype, frac = d

            if dtype != "data":
                continue

            if isinstance(frac, str):
                data_str = frac
            else:
                assert "object" in frac and frac["object"] == "text_completion"
                data_str = frac["choices"][0]["text"]  # type: ignore

            data += data_str

        try:
            out.append(decoder.decode(data))
        except json.JSONDecodeError:
            fails += 1

        if fails >= MAX_FAILS and llm["model_type"] == "or":
            logger.error(
                f"Sampling failed {fails} times for sample {i+1}. Aborting further sampling."
            )
            raise RuntimeError("Maximum sampling failures reached.")

    stop.set()
    for t in _ctx["t"]:
        t.join()

    df = pd.DataFrame({"entity": map(str, out)})
    return {
        "ids": pd.DataFrame({"id": df.index}),
        "data": df,
    }


def sample(
    gen,
    prompt: str,
    counts,
    meta: Any,
    pgm_samples: pd.DataFrame,
    ref: pd.DataFrame,
    data: dict[str, LazyDataset],
):

    ctx = {
        "t": [],
        "stop": threading.Event(),
    }

    try:
        return _sample(
            gen,
            prompt,
            counts,
            meta,
            pgm_samples,
            ref,
            data,
            ctx,
        )
    finally:
        ctx["stop"].set()
        for t in ctx["t"]:
            ctx["t"].join()
