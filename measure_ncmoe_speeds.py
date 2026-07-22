#!/usr/bin/env python3
"""Sweep -ncmoe values for llama-server and record context/VRAM/throughput.

For each N in NCMOE_VALUES: start llama-server, wait for it to become
healthy, read n_ctx from /props, measure VRAM and server RSS, run a
warmup request followed by a measured one, and append a row to the CSV.
Configs that fail to load within LOAD_TIMEOUT_S are recorded as failed.
"""

import csv
import subprocess
import sys
import time
from pathlib import Path

import requests

# --- Configuration -----------------------------------------------------------

MODEL = "unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_NL_XL"
PORT = 8000
BASE_URL = f"http://localhost:{PORT}"
NCMOE_VALUES = list(range(41))
LOAD_TIMEOUT_S = 300  # generous: HF cache hit is fast, cold download is not
N_PREDICT = 6144
CSV_PATH = Path("sweep_results.csv")


AUSTEN = Path("austen.txt").read_text()

USER_MSG = Path("user_message.txt").read_text() if Path("user_message.txt").exists() else (
    "Compose a poem -- a poem about a haircut! But lofty, tragic, timeless, "
    "full of love, treachery, retribution, quiet heroism in the face of certain "
    "doom! Six lines, cleverly rhymed, and every word beginning with the letter S!"
)

PROMPT = (
    "<|im_start|>system\n"
    "You are a helpful assistant<|im_end|>\n"
    "<|im_start|>user\n"
    f"What do you think of this text? {AUSTEN}<|im_end|>\n"
    "<|im_start|>assistant\n"
    "That is the start of *Pride and Prejudice* by Jane Austen, and I think it is excellent.<|im_end|>\n"
    "<|im_start|>user\n"
    f"{USER_MSG}<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n"
)


SERVER_CMD = [
    "llama-server",
    "-hf", MODEL,
    "--jinja",
    "-ngl", "all",
    "-fa", "on",
    "-sm", "none",
    "--temp", "0.6",
    "--top-k", "20",
    "--top-p", "0.95",
    "--min-p", "0",
    "--presence-penalty", "1.5",
    "--no-context-shift",
    "--port", str(PORT),
]

# --- Helpers -----------------------------------------------------------------


def wait_for_health(timeout_s: float) -> bool:
    """Poll /health until the server reports ready, or time out."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                return True
        except requests.ConnectionError:
            pass  # server not up yet
        time.sleep(2)
    return False


def get_n_ctx() -> int | None:
    r = requests.get(f"{BASE_URL}/props", timeout=10)
    r.raise_for_status()
    props = r.json()
    # Location has moved between builds; check the likely spots.
    for path in (
        ("default_generation_settings", "n_ctx"),
        ("n_ctx",),
    ):
        node = props
        try:
            for key in path:
                node = node[key]
            return int(node)
        except (KeyError, TypeError):
            continue
    return None


def server_vram_mib(pid: int) -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader,nounits"],
        text=True,
    )
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        p, mem = (x.strip() for x in line.split(","))
        if int(p) == pid:
            return int(mem)
    raise Exception(f"Could not find process {pid} in output!")


def server_rss_mib(pid: int) -> int:
    out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True)
    return int(out.strip()) // 1024


def run_completion(n_predict: int) -> dict:
    """Fire a completion at the native endpoint and return its timings."""
    r = requests.post(
        f"{BASE_URL}/completion",
        json={
            "prompt": PROMPT,
            "n_predict": n_predict,
            "temperature": 0.6,
            "cache_prompt": False,
        },
        timeout=450,
    )
    r.raise_for_status()
    return r.json().get("timings", {})


def stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


# --- Main sweep --------------------------------------------------------------


def main() -> None:
    new_file = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow([
                "ncmoe", "loaded", "n_ctx", "vram_mib", "rss_mib",
                "prompt_n", "prompt_tps", "gen_n", "gen_tps",
            ])

        for n in NCMOE_VALUES:
            print(f"=== -ncmoe {n} ===")
            cmd = SERVER_CMD + ["-ncmoe", str(n)]
            proc = subprocess.Popen(
                cmd,
            )
            try:
                if not wait_for_health(LOAD_TIMEOUT_S):
                    print("  did not become healthy -- recording as failed")
                    writer.writerow([n, "no"] + [""] * 7)
                    f.flush()
                    continue

                n_ctx = get_n_ctx()
                vram = server_vram_mib(proc.pid)
                rss = server_rss_mib(proc.pid)
                print(f"  n_ctx={n_ctx}  vram={vram} MiB  rss={rss} MiB")

                print("  warmup request...")
                run_completion(n_predict=32)

                print("  measured request...")
                t = run_completion(n_predict=N_PREDICT)
                prompt_tps = t.get("prompt_per_second")
                gen_tps = t.get("predicted_per_second")
                print(f"  prompt: {prompt_tps or 0:.1f} t/s   gen: {gen_tps or 0:.1f} t/s")

                writer.writerow([
                    n, "yes", n_ctx, vram, rss,
                    t.get("prompt_n"), round(prompt_tps or 0, 1),
                    t.get("predicted_n"), round(gen_tps or 0, 1),
                ])
                f.flush()
            except Exception as e:  # noqa: BLE001 -- keep the sweep going
                print(f"  error: {e}", file=sys.stderr)
                writer.writerow([n, "error"] + [""] * 7)
                f.flush()
            finally:
                stop_server(proc)
                time.sleep(5)  # let VRAM actually drain before the next run

    print(f"\nDone. Results in {CSV_PATH}")


if __name__ == "__main__":
    main()
