import logging
import threading
import time
from typing import Any, Literal, Mapping, Type, TypedDict

import numpy as np
import pandas as pd
from pydantic.main import BaseModel

from pasteur.utils import LazyDataset
from pasteur.utils.progress import prange

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


class EvalOutputType(BaseModel):
    score: Literal[1, 2, 3, 4, 5]


class EvalOutputReasonType(BaseModel):
    reasoning: str
    score: Literal[1, 2, 3, 4, 5]


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


gpu_lock = threading.Lock()


def _gpu_monitor_worker(name: str, stop: threading.Event, run_id: str):
    import subprocess
    import csv
    import tempfile

    process = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,power.draw,power.draw.average,power.draw.instant,utilization.gpu,utilization.memory,memory.used,memory.total",
            "--format=csv,nounits",
            "-l=1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    with (
        tempfile.TemporaryDirectory() as tmpdir,
        open(f"{tmpdir}/gpu_{name}.csv", "w", newline="") as tmpfile,
    ):
        writer = csv.writer(tmpfile)

        while not stop.is_set():
            line = process.stdout.readline()
            if not line:
                continue
            writer.writerow(line.strip().split(", "))

        process.terminate()
        process.wait()
        tmpfile.flush()

        try:
            import mlflow

            if mlflow.active_run() is None:
                mlflow.start_run(run_id=run_id)
            mlflow.log_artifact(tmpfile.name, artifact_path=f"_raw/energy")
        except Exception:
            logger.error("Error logging GPU energy info to MLflow.", exc_info=True)


class hold_gpu_lock:
    def __init__(self, name: str | None = None):
        self.name = name
        self.t = None
        self.stop_event = threading.Event()

    def __enter__(self):
        gpu_lock.acquire()

        try:
            import mlflow

            ar = mlflow.active_run()
            assert ar is not None, "MLflow active run is required for GPU monitoring."
            run_id = ar.info.run_id
            self.t = threading.Thread(
                target=_gpu_monitor_worker,
                args=(self.name or "unknown", self.stop_event, run_id),
                daemon=True,
            )
            self.t.start()
        except Exception:
            logger.error("Error starting GPU monitor thread.", exc_info=True)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.t is not None:
            self.stop_event.set()
            self.t.join()
        gpu_lock.release()


def load_llm_model(
    params: AmalgamHFParams | AmalgamORParams,
    output_type,
):
    llms = []

    logger.info(f"Loading LLM model for sampling: {params}")

    for _ in range(params.get("workers", 1)):
        llms.append(_load_llm_model(params, output_type))

    return {
        "type": "generate",
        "model_type": params["type"],
        "llms": llms,
    }


def load_llm_model_eval(
    params: AmalgamHFParams | AmalgamORParams,
    reason: bool = True,
):
    llms = []

    logger.info(f"Loading LLM model for evaluation: {params}")

    for _ in range(params.get("workers", 1)):
        llms.append(
            _load_llm_model(params, EvalOutputReasonType if reason else EvalOutputType)
        )

    return {
        "type": "evaluate",
        "model_type": params["type"],
        "llms": llms,
    }


def get_or_api_key() -> str:
    from pasteur.kedro import context

    assert context is not None, "Kedro context is not initialized."
    return context._get_config_credentials()["openrouter"]


def _printer(prompt, sample_num, sample_n, q, stop, task):
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
        if stop.is_set():
            break

        dtype, j = token

        end = dtype is None
        if not end:
            if dtype is None:
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
                f":ephemeral:{task} {sample_num}/{sample_n}. Prompt: {prompt_reduced}{thought_str}\nData:\n{pretty}"
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
    task,
):
    import queue

    while not stop.is_set():
        failed = True
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
                target=_printer,
                args=(prompt, sample_num, sample_n, pq, stop, task),
                daemon=True,
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

            failed = stop.is_set()
        except Exception:
            import traceback

            logger.error(f"Error in thought worker:\n{traceback.format_exc()}")
        finally:
            end = time.perf_counter()
            out_q.put((start, ttft_thought, ttft, end, data, failed))

            pq.put((None, None))
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
                "Sampling Entity",
            ),
            daemon=True,
        )
        t.start()
        _ctx["t"].append(t)

    prompts = []
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
        prompts.append(fprompt)

    # Grab energy info
    tracker = CacheTracker()
    # FIXME: Generalize this
    tokenizer = gen["llms"][0]["llm"].tokenizer
    cached_tokens = 0
    input_tokens = 0
    output_tokens = 0

    in_time = 0
    out_time = 0

    for i in prange(n_samples, desc="Processing entities"):
        start, ttft_thought, ttft, end, chunks, failed = out_q.get()

        prompt = prompts[i]
        ptokens = tokenizer.encode(prompt)[0]
        ctokens = tracker.get_cached_len(ptokens)
        cached_tokens += ctokens
        input_tokens += len(ptokens) - ctokens

        in_time += (ttft if ttft is not None else start) - start
        out_time += end - (ttft if ttft is not None else start)

        if not chunks:
            continue

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

        otokens = tokenizer.encode(data)[0]
        output_tokens += len(otokens)
        tracker.add_cached_tokens(ptokens + otokens)

        if not failed:
            try:
                out.append(decoder.decode(data))
            except json.JSONDecodeError:
                fails += 1
        else:
            fails += 1

        if fails >= MAX_FAILS and llm["model_type"] == "or":
            logger.error(
                f"Sampling failed {fails} times for sample {i+1}. Aborting further sampling."
            )
            raise RuntimeError("Maximum sampling failures reached.")

    stop.set()
    for t in _ctx["t"]:
        t.join()

    try:
        input_tps = input_tokens / in_time if in_time > 0 else 0
        output_tps = output_tokens / out_time if out_time > 0 else 0

        logger.info(
            f"""\
Entities Generated: {len(out)}, failed: {fails}, total: {n_samples}.

# Token information
Cached: {cached_tokens:12,d}
 Input: {input_tokens:12,d}
Output: {output_tokens:12,d}
 Total: {input_tokens + output_tokens:12,d}

# Time spent
 Input time: {in_time if in_time else float('NaN'):7,.2f} s
Output time: {out_time if out_time else float('NaN'):7,.2f} s
 Total time: {in_time + out_time if in_time + out_time else float('NaN'):7,.2f} s

# Throughput
 Input tokens per second: {input_tps if input_tps else float('NaN'):8,.2f} t/s
Output tokens per second: {output_tps if output_tps else float('NaN'):8,.2f} t/s
"""
        )

        import mlflow

        if mlflow.active_run() is not None:
            mlflow.log_param("sampling.cached_tokens", cached_tokens)
            mlflow.log_param("sampling.input_tokens", input_tokens)
            mlflow.log_param("sampling.input_time", in_time)
            mlflow.log_param("sampling.input_tps", input_tps)
            mlflow.log_param("sampling.output_tokens", output_tokens)
            mlflow.log_param("sampling.output_time", out_time)
            mlflow.log_param("sampling.output_tps", output_tps)
            mlflow.log_param("sampling.sample_n", len(out))
            mlflow.log_param("sampling.failures", fails)
    except Exception:
        logger.error("Error logging sampling performance to MLflow.", exc_info=True)

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
            t.join()


def _evaluate(
    gen,
    prompt: str,
    counts,
    wrk_flat: pd.DataFrame,
    wrk_json: dict[str, LazyDataset],
    eval_flat: pd.DataFrame,
    eval_json: dict[str, LazyDataset],
    max_samples: int | None,
    top_k: int,
    split: str,
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

    for i, (k, v) in enumerate(wrk_json["ids"].items()):
        part_map[i] = k
        uv = v()
        id_map.append(uv.rename(columns={next(iter(uv.columns)): "id"}).assign(part=i))

    id_map = pd.concat(id_map).set_index("id")

    id_map_eval = []
    part_map_eval = {}

    for i, (k, v) in enumerate(eval_json["ids"].items()):
        part_map_eval[i] = k
        uv = v()
        id_map_eval.append(
            uv.rename(columns={next(iter(uv.columns)): "id"}).assign(part=i)
        )

    id_map_eval = pd.concat(id_map_eval).set_index("id")

    if max_samples is not None:
        eval_flat = eval_flat.iloc[:max_samples]

    for p in range(0, len(wrk_flat) // PART_SIZE + 1):
        start = p * PART_SIZE
        end = min((p + 1) * PART_SIZE, len(wrk_flat))

        scores = np.zeros((end - start, len(eval_flat)), np.uint16)

        for k, v in norm_scores.items():
            mult = 5 if "_total_count" in k else 1

            vals = v[wrk_flat.iloc[start:end][k].values[:, None]]
            eqs = (
                wrk_flat.iloc[start:end][k].values[:, None]
                == eval_flat.loc[:, k].values[None, :]
            )
            scores += mult * (vals * eqs).astype(np.uint16)

        idx = scores.argsort(axis=0)[-top_k:, :]
        scores_topk = np.take_along_axis(scores, idx, axis=0)
        idx_index = np.take_along_axis(
            np.array(wrk_flat.index), start + idx.flatten()
        ).reshape(idx.shape)

        topk_idx.append(idx_index)
        topk_scores.append(scores_topk)

    full_idx = np.concat(topk_idx)
    full_scores = np.concat(topk_scores)

    lookups = np.take_along_axis(
        full_idx, full_scores.argsort(axis=0)[-top_k:, :], axis=0
    )

    t = None
    stop = _ctx["stop"]

    jdata = wrk_json["data"]
    jdata_eval = eval_json["data"]
    n_samples = eval_flat.shape[0]

    fails = 0

    in_q = queue.Queue()
    out_q = queue.Queue()

    for i, llm in enumerate(gen["llms"]):
        t = threading.Thread(
            target=_worker,
            args=(
                llm["generator"],
                None,  # llm["generator_thought"],
                stop,
                in_q,
                out_q,
                n_samples,
                False,
                i == 0,
                f"Evaluating {split} Entity",
            ),
            daemon=True,
        )
        t.start()
        _ctx["t"].append(t)

    prompts = []
    for i in range(n_samples):
        samples = []
        for k in lookups[:, i].tolist():
            arr = jdata[part_map[int(id_map.loc[k].iloc[0])]]()
            val = arr.loc[k, next(iter(arr.columns))]
            samples.append(str(val))

        samples = "\n".join(samples)

        arr = jdata_eval[
            part_map_eval[int(id_map_eval.loc[eval_flat.index[i]].iloc[0])]
        ]()
        val = arr.loc[eval_flat.index[i], next(iter(arr.columns))]
        eval_str = str(val)

        fprompt = (
            prompt.replace("<eval>", eval_str)
            .replace("<samples>", samples)
            .replace("<samples_n>", str(top_k))
        )

        sample_num = i + 1
        in_q.put((fprompt, sample_num))
        prompts.append(fprompt)

    # Grab energy info
    tracker = CacheTracker()
    # FIXME: Generalize this
    tokenizer = gen["llms"][0]["llm"].tokenizer
    cached_tokens = 0
    input_tokens = 0
    output_tokens = 0

    in_time = 0
    out_time = 0

    for i in prange(n_samples, desc=f"Evaluating {split.capitalize()} entities"):
        start, ttft_thought, ttft, end, chunks, failed = out_q.get()

        prompt = prompts[i]
        ptokens = tokenizer.encode(prompt)[0]
        ctokens = tracker.get_cached_len(ptokens)
        cached_tokens += ctokens
        input_tokens += len(ptokens) - ctokens

        in_time += (ttft if ttft is not None else start) - start
        out_time += end - (ttft if ttft is not None else start)

        if not chunks:
            continue

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

        otokens = tokenizer.encode(data)[0]
        output_tokens += len(otokens)
        tracker.add_cached_tokens(ptokens + otokens)

        if not failed:
            try:
                out.append(decoder.decode(data))
            except json.JSONDecodeError:
                fails += 1
        else:
            fails += 1

        if fails >= MAX_FAILS and llm["model_type"] == "or":
            logger.error(
                f"Sampling failed {fails} times for sample {i+1}. Aborting further sampling."
            )
            raise RuntimeError("Maximum sampling failures reached.")

    stop.set()
    for t in _ctx["t"]:
        t.join()

    try:
        input_tps = input_tokens / in_time if in_time > 0 else 0
        output_tps = output_tokens / out_time if out_time > 0 else 0

        logger.info(
            f"""\
{split.capitalize()} Entities evaluated: {len(out)}, failed: {fails}, total: {n_samples}.

# Token information
Cached: {cached_tokens:12,d}
 Input: {input_tokens:12,d}
Output: {output_tokens:12,d}
 Total: {input_tokens + output_tokens:12,d}

# Time spent
 Input time: {in_time if in_time else float('NaN'):7,.2f} s
Output time: {out_time if out_time else float('NaN'):7,.2f} s
 Total time: {in_time + out_time if in_time + out_time else float('NaN'):7,.2f} s

# Throughput
 Input tokens per second: {input_tps if input_tps else float('NaN'):8,.2f} t/s
Output tokens per second: {output_tps if output_tps else float('NaN'):8,.2f} t/s
"""
        )

        import mlflow

        if mlflow.active_run() is not None:
            mlflow.log_param(f"eval.{split}.cached_tokens", cached_tokens)
            mlflow.log_param(f"eval.{split}.input_tokens", input_tokens)
            mlflow.log_param(f"eval.{split}.input_time", in_time)
            mlflow.log_param(f"eval.{split}.input_tps", input_tps)
            mlflow.log_param(f"eval.{split}.output_tokens", output_tokens)
            mlflow.log_param(f"eval.{split}.output_time", out_time)
            mlflow.log_param(f"eval.{split}.output_tps", output_tps)
            mlflow.log_param(f"eval.{split}.sample_n", len(out))
            mlflow.log_param(f"eval.{split}.failures", fails)
    except Exception:
        logger.error("Error logging sampling performance to MLflow.", exc_info=True)

    return out


def evaluate(
    gen,
    prompt: str,
    counts,
    wrk_flat: pd.DataFrame,
    wrk_json: dict[str, LazyDataset],
    eval_flat: pd.DataFrame,
    eval_json: dict[str, LazyDataset],
    max_samples: int | None = None,
    top_k: int = 3,
    split: str = "ref",
):

    ctx = {
        "t": [],
        "stop": threading.Event(),
    }

    try:
        return _evaluate(
            gen,
            prompt,
            counts,
            wrk_flat,
            wrk_json,
            eval_flat,
            eval_json,
            max_samples,
            top_k,
            split,
            ctx,
        )
    finally:
        ctx["stop"].set()
        for t in ctx["t"]:
            t.join()
