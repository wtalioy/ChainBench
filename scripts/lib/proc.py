"""Subprocess helpers."""

from __future__ import annotations

import os
import re
import codecs
import subprocess
import sys
from pathlib import Path
from typing import Callable

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def run_command(
    command: list[str],
    cwd: Path | None = None,
    timeout_sec: int | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        full_env = os.environ.copy()
        if env:
            full_env.update(env)
        return subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            env=full_env,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return subprocess.CompletedProcess(
            args=command,
            returncode=124,
            stdout=stdout,
            stderr=(stderr + f"\nTIMEOUT after {timeout_sec}s").strip(),
        )


def run_command_streaming(
    command: list[str],
    cwd: Path,
    log_path: Path,
    on_line: Callable[[str], None] | None = None,
    env: dict[str, str] | None = None,
    tee_output: bool = False,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("COMMAND:\n" + " ".join(command) + "\n\nOUTPUT:\n")
        log_handle.flush()

        full_env = os.environ.copy()
        if env:
            full_env.update(env)
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=full_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        assert process.stdout is not None
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        line_buffer = ""
        last_logged_line = ""

        def flush_line() -> None:
            nonlocal line_buffer, last_logged_line
            cleaned = ANSI_ESCAPE_RE.sub("", line_buffer).rstrip()
            if cleaned:
                if cleaned != last_logged_line:
                    log_handle.write(cleaned + "\n")
                    log_handle.flush()
                    if on_line:
                        on_line(cleaned)
                    last_logged_line = cleaned
            line_buffer = ""

        while True:
            chunk = process.stdout.read(1)
            if chunk == b"":
                break
            text_chunk = decoder.decode(chunk)
            if not text_chunk:
                continue
            for char in text_chunk:
                if tee_output:
                    sys.stdout.write(char)
                    sys.stdout.flush()
                if char == "\r":
                    # Progress bars often advance via carriage returns without a
                    # trailing newline. Flush the current snapshot so long-running
                    # jobs write live status updates into the log file too.
                    flush_line()
                    continue
                if char == "\n":
                    flush_line()
                else:
                    line_buffer += char
        remaining = decoder.decode(b"", final=True)
        if remaining:
            line_buffer += remaining
        flush_line()
        return process.wait()
