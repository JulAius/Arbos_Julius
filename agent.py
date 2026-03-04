import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

PROMPT_FILE = Path(__file__).parent / "PROMPT.md"
WORKING_DIR = Path(__file__).parent
HISTORY_DIR = WORKING_DIR / "history"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("agent-loop")


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


def load_prompt() -> str:
    if not PROMPT_FILE.exists():
        log.error("Prompt file not found: %s", PROMPT_FILE)
        sys.exit(1)
    text = PROMPT_FILE.read_text().strip()
    if not text:
        log.error("Prompt file is empty: %s", PROMPT_FILE)
        sys.exit(1)
    return text


def make_run_dir() -> Path:
    HISTORY_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = HISTORY_DIR / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _describe_tool_call(tc: dict) -> str:
    """Extract a short human-readable description from a stream-json tool_call object."""
    for key, val in tc.items():
        if not isinstance(val, dict):
            continue
        args = val.get("args", {})
        if "path" in args:
            return f"{key}({args['path']})"
        if "command" in args:
            cmd = args["command"]
            return f"{key}({cmd[:80]}{'…' if len(cmd) > 80 else ''})"
        if "pattern" in args:
            return f"{key}(pattern={args['pattern']!r})"
        arg_summary = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:2])
        return f"{key}({arg_summary})"
    return str(list(tc.keys()))


def run_agent(cmd: list[str], phase: str, output_file: Path) -> subprocess.CompletedProcess:
    stream_cmd = []
    for arg in cmd:
        if arg == "--output-format":
            stream_cmd.append(arg)
            continue
        if stream_cmd and stream_cmd[-1] == "--output-format":
            stream_cmd.append("stream-json")
            continue
        stream_cmd.append(arg)
    if "--stream-partial-output" not in stream_cmd:
        stream_cmd.insert(-1, "--stream-partial-output")

    log.info("Running: %s", " ".join(stream_cmd[:6]) + (" ..." if len(stream_cmd) > 6 else ""))
    t0 = time.monotonic()

    proc = subprocess.Popen(
        stream_cmd, cwd=WORKING_DIR,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )

    result_text = ""
    raw_lines: list[str] = []
    for line in iter(proc.stdout.readline, ""):
        raw_lines.append(line)
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue

        etype = evt.get("type")
        subtype = evt.get("subtype")

        if etype == "tool_call" and subtype == "started":
            desc = _describe_tool_call(evt.get("tool_call", {}))
            log.info("[%s] tool call: %s", phase, desc)
        elif etype == "tool_call" and subtype == "completed":
            desc = _describe_tool_call(evt.get("tool_call", {}))
            log.info("[%s] tool done: %s", phase, desc)
        elif etype == "assistant":
            text = ""
            for block in evt.get("message", {}).get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    text += block.get("text", "")
            if text.strip():
                for tline in text.strip().splitlines():
                    log.info("[%s] %s", phase, tline)
        elif etype == "result":
            result_text = evt.get("result", "")
            dur = evt.get("duration_ms", 0)
            usage = evt.get("usage", {})
            log.info(
                "[%s] done  duration=%s  tokens_in=%s  tokens_out=%s",
                phase, fmt_duration(dur / 1000),
                usage.get("inputTokens", "?"), usage.get("outputTokens", "?"),
            )

    returncode = proc.wait()
    elapsed = time.monotonic() - t0
    output_file.write_text("".join(raw_lines))
    log.info(
        "%s finished  rc=%d  duration=%s",
        phase, returncode, fmt_duration(elapsed),
    )

    return subprocess.CompletedProcess(
        args=cmd, returncode=returncode,
        stdout=result_text, stderr="",
    )


def extract_text(result: subprocess.CompletedProcess) -> str:
    output = result.stdout or ""
    if not output.strip():
        output = result.stderr or "(no output)"
    return output


def run_step(prompt: str) -> None:
    run_dir = make_run_dir()
    t0 = time.monotonic()

    log_file = run_dir / "logs.txt"
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    log.addHandler(file_handler)

    log.info("Run dir: %s", run_dir)
    log.info("Log file: %s", log_file)

    # --- Phase 1: Plan ---
    log.info("=" * 60)
    log.info("Planning phase")
    log.info("=" * 60)
    preview = prompt[:200] + ("…" if len(prompt) > 200 else "")
    log.info("Prompt preview: %s", preview)

    plan_result = run_agent(
        ["agent", "-p", "--force", "--mode", "plan", "--output-format", "text", prompt],
        phase="plan",
        output_file=run_dir / "plan_output.txt",
    )

    plan_text = extract_text(plan_result)
    (run_dir / "plan.md").write_text(plan_text)
    log.info("Plan saved → %s (%d chars)", run_dir / "plan.md", len(plan_text))

    if plan_result.returncode != 0:
        log.warning("Plan phase exited with code %d — skipping execution", plan_result.returncode)
        log.removeHandler(file_handler)
        file_handler.close()
        return

    # --- Phase 2: Execute ---
    log.info("=" * 60)
    log.info("Execution phase")
    log.info("=" * 60)

    execute_prompt = (
        f"Here is the plan that was previously generated:\n\n"
        f"---\n{plan_text}\n---\n\n"
        f"Now implement this plan. The original request was:\n\n{prompt}"
    )
    log.info("Execution prompt size: %d chars (plan=%d + original=%d)", len(execute_prompt), len(plan_text), len(prompt))

    exec_result = run_agent(
        ["agent", "-p", "--force", "--output-format", "text", execute_prompt],
        phase="exec",
        output_file=run_dir / "exec_output.txt",
    )

    exec_text = extract_text(exec_result)
    (run_dir / "rollout.md").write_text(exec_text)
    log.info("Rollout saved → %s (%d chars)", run_dir / "rollout.md", len(exec_text))

    elapsed = time.monotonic() - t0
    if exec_result.returncode != 0:
        log.warning("Execution phase exited with code %d", exec_result.returncode)
    else:
        log.info("Run completed successfully")
    log.info("Total duration: %s", fmt_duration(elapsed))

    log.removeHandler(file_handler)
    file_handler.close()


def main() -> None:
    log.info("Prompt file : %s", PROMPT_FILE)
    log.info("Working dir : %s", WORKING_DIR)
    log.info("History dir : %s", HISTORY_DIR)

    loop_count = 0
    while True:
        loop_count += 1
        prompt = load_prompt()
        log.info("Loop iteration %d  prompt=%d chars", loop_count, len(prompt))
        run_step(prompt)


if __name__ == "__main__":
    main()
