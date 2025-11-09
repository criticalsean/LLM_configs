#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ollama ↔ Filesystem bridge (two roots + read/write)
- Allowed roots:
    /Users/seanbenson/SaltStackAltDoor
    /Users/seanbenson/AIfiles
- Supports: [[LS]], [[LSR]], [[STAT]], [[CAT]], [[WRITE]], [[APPEND]], [[DELETE]]
"""

import argparse
import json
import os
import re
import pathlib
import stat
import urllib.request
from typing import Optional, Tuple

# -------------------- Config --------------------
ALLOWED_ROOTS = [
    pathlib.Path("/Users/seanbenson/SaltStackAltDoor").resolve(),
    pathlib.Path("/Users/seanbenson/AIfiles").resolve(),
]
DEFAULT_ROOT = ALLOWED_ROOTS[0]          # non-absolute paths are relative to this root
MAX_BYTES = 2_000_000
LS_MAX = 1000
FOLLOW_SYMLINKS = False
OVERWRITE_OK = True
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
API_CHAT = f"{OLLAMA_HOST}/api/chat"
# ------------------------------------------------

# -------------------- Directive Patterns --------------------
RE_LS    = re.compile(r'^\s*\[\[LS:(?P<path>/.+?)\]\]\s*$')
RE_LSR   = re.compile(r'^\s*\[\[LSR:(?P<path>/.+?):(?P<depth>\d+)\]\]\s*$')
RE_CAT   = re.compile(r'^\s*\[\[CAT:(?P<path>/.+?)(?::(?P<start>\d+)-(?P<end>\d+))?\]\]\s*$')
RE_STAT  = re.compile(r'^\s*\[\[STAT:(?P<path>/.+?)\]\]\s*$')
RE_WRITE = re.compile(r'^\s*\[\[WRITE:(?P<path>/.+?)\]\]\s*(?P<data>.*)\Z', re.DOTALL)
RE_APPEND= re.compile(r'^\s*\[\[APPEND:(?P<path>/.+?)\]\]\s*(?P<data>.*)\Z', re.DOTALL)
RE_DELETE= re.compile(r'^\s*\[\[DELETE:(?P<path>/.+?)\]\]\s*$', re.DOTALL)
# ------------------------------------------------------------

def roots_list_for_prompt():
    return "\n- ".join(str(r) for r in ALLOWED_ROOTS)

SYSTEM_PROMPT = f"""You are connected to a filesystem bridge.

Roots (you may access only under these):
- {roots_list_for_prompt()}

Capabilities:
- List: [[LS:/abs/path]] or [[LSR:/abs/path:depth]]
- Read: [[CAT:/abs/path:start-end]] or [[CAT:/abs/path]]
- Stat: [[STAT:/abs/path]]
- Write: [[WRITE:/abs/path]] then include the file text on following lines
- Append: [[APPEND:/abs/path]] then include the text to append
- Delete: [[DELETE:/abs/path]]

Contract:
- Output exactly ONE bracketed command when you need filesystem data or actions, then WAIT.
- Do NOT claim file contents unless they appear from a bridge reply in this chat.
- If bridge returns NOT FOUND or PERMISSION DENIED, adjust and try up to 2 times.
- If you can answer without FS access, answer directly.
- Keep internal reasoning private; output only actions, final answers, or brief justifications.

Diagnostics:
- Assume the bridge is active if you see BRIDGE_READY=1.

Output rules:
- Acting: output only the bracketed command on a single line (plus body for WRITE/APPEND).
- Answering: cite the path (and byte range if used).

Notes:
- Use absolute paths to pick a root explicitly. If you provide a relative path, it will be interpreted under {DEFAULT_ROOT}.
"""

FEWSHOTS = [
    {"role":"user","content":"List the main root."},
    {"role":"assistant","content":f"[[LS:{DEFAULT_ROOT}]]"},
    {"role":"assistant","content":"(example) I see items under the default root. Want me to open one?"},
    {"role":"user","content":"Also list the AIfiles root."},
    {"role":"assistant","content":f"[[LS:{ALLOWED_ROOTS[1]}]]"},
    {"role":"assistant","content":"(example) Items under AIfiles listed."},
]

# -------------------- Ollama call --------------------
def call_ollama(model, messages, stream=False, options=None):
    payload = {"model": model, "messages": messages, "stream": stream}
    if options:
        payload["options"] = options
    req = urllib.request.Request(
        API_CHAT,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())

# -------------------- Path guards --------------------
def _is_within(root: pathlib.Path, target: pathlib.Path) -> bool:
    root_str = str(root) + os.sep
    targ_str = str(target)
    return targ_str == str(root) or targ_str.startswith(root_str)

def ensure_in_roots(user_path: str) -> pathlib.Path:
    """
    Resolve 'user_path' safely under one of ALLOWED_ROOTS.
    - Absolute paths: must already be inside an allowed root; otherwise we try
      interpreting '/something' as root-relative under each allowed root.
    - Relative paths: resolved under DEFAULT_ROOT.
    Blocks symlink targets (final path) unless FOLLOW_SYMLINKS=True.
    """
    p = pathlib.Path(user_path)

    candidates = []

    if p.is_absolute():
        # 1) If already under any root, accept as-is
        real = p.resolve(strict=False)
        for root in ALLOWED_ROOTS:
            if _is_within(root, real):
                candidates.append(real)
        # 2) Otherwise, interpret as root-relative ('/sub/dir' inside each root)
        if not candidates:
            try:
                rel = p.relative_to("/")  # drop leading slash
            except Exception:
                rel = p
            for root in ALLOWED_ROOTS:
                candidates.append((root / rel).resolve(strict=False))
    else:
        candidates.append((DEFAULT_ROOT / p).resolve(strict=False))

    # Pick the first candidate that is within some root (after resolving)
    for real in candidates:
        for root in ALLOWED_ROOTS:
            if _is_within(root, real):
                # Symlink block at final target
                try:
                    if real.is_symlink() and not FOLLOW_SYMLINKS:
                        raise PermissionError(f"PERMISSION DENIED (symlink): {user_path}")
                except Exception:
                    pass
                return real

    raise PermissionError(f"PERMISSION DENIED: outside allowed roots: {user_path}")

# -------------------- FS Operations --------------------
def do_ls(path: str):
    try:
        rp = ensure_in_roots(path)
        if not rp.exists():
            return f"NOT FOUND: {path}"
        if rp.is_file():
            return json.dumps([str(rp)], indent=2)
        items = []
        for i, entry in enumerate(sorted(rp.iterdir(), key=lambda e: e.name)):
            if i >= LS_MAX: break
            t = "dir" if entry.is_dir() else "file"
            items.append({"name": entry.name, "type": t})
        return json.dumps(items, indent=2)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"

def do_lsr(path: str, depth: int):
    try:
        rp = ensure_in_roots(path)
        if not rp.exists():
            return f"NOT FOUND: {path}"
        out, count = [], 0
        def walk(p: pathlib.Path, d: int):
            nonlocal count
            if count >= LS_MAX: return
            try:
                for entry in sorted(p.iterdir(), key=lambda e: e.name):
                    if count >= LS_MAX: return
                    rel = str(entry.relative_to(rp)) if p != rp else entry.name
                    t = "dir" if entry.is_dir() else "file"
                    out.append({"path": rel, "type": t})
                    count += 1
                    if entry.is_dir() and d > 0:
                        walk(entry, d - 1)
            except PermissionError:
                out.append({"path": str(p), "type": "dir", "error": "PERMISSION DENIED"})
        if rp.is_file():
            out.append({"path": rp.name, "type": "file"})
        else:
            walk(rp, depth)
        return json.dumps(out, indent=2)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"

def do_stat(path: str):
    try:
        rp = ensure_in_roots(path)
        if not rp.exists():
            return f"NOT FOUND: {path}"
        st = rp.stat()
        info = {
            "path": str(rp),
            "type": "dir" if rp.is_dir() else "file",
            "size": st.st_size,
            "mode": stat.filemode(st.st_mode),
            "mtime": int(st.st_mtime),
            "ctime": int(st.st_ctime),
        }
        return json.dumps(info, indent=2)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"

def do_cat(path: str, start=None, end=None):
    try:
        rp = ensure_in_roots(path)
        if not rp.exists(): return f"NOT FOUND: {path}"
        if rp.is_dir(): return f"ERROR: Is a directory: {path}"
        if start is None and end is None:
            size = rp.stat().st_size
            if size > MAX_BYTES:
                return f"ERROR: File too large ({size} bytes > {MAX_BYTES})."
            with open(rp, "rb") as f: data = f.read(MAX_BYTES)
            return data.decode(errors="replace")
        s = int(start or 0)
        e = int(end or min(rp.stat().st_size, s + MAX_BYTES))
        if e - s > MAX_BYTES: e = s + MAX_BYTES
        with open(rp, "rb") as f:
            f.seek(s)
            data = f.read(e - s)
        return data.decode(errors="replace")
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"

def do_write(path: str, data: str):
    try:
        rp = ensure_in_roots(path)
        if rp.exists() and not OVERWRITE_OK:
            return f"ERROR: File already exists: {rp}"
        if rp.is_dir(): return f"ERROR: Cannot write to directory: {path}"
        rp.parent.mkdir(parents=True, exist_ok=True)
        with open(rp, "w", encoding="utf-8") as f:
            f.write(data or "")
        return f"WRITE OK: {rp}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"

def do_append(path: str, data: str):
    try:
        rp = ensure_in_roots(path)
        if rp.is_dir(): return f"ERROR: Cannot append to directory: {path}"
        rp.parent.mkdir(parents=True, exist_ok=True)
        with open(rp, "a", encoding="utf-8") as f:
            f.write(data or "")
        return f"APPEND OK: {rp}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"

def do_delete(path: str):
    try:
        rp = ensure_in_roots(path)
        if not rp.exists(): return f"NOT FOUND: {path}"
        if rp.is_dir(): return f"ERROR: {path} is a directory (non-recursive delete refused)"
        rp.unlink()
        return f"DELETE OK: {rp}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"

# -------------------- Directive Parser --------------------
def extract_directive(text: str):
    for regex, kind in (
        (RE_LS, "LS"),
        (RE_LSR, "LSR"),
        (RE_CAT, "CAT"),
        (RE_STAT, "STAT"),
        (RE_WRITE, "WRITE"),
        (RE_APPEND, "APPEND"),
        (RE_DELETE, "DELETE"),
    ):
        m = regex.match(text or "")
        if m:
            return kind, m
    return None, None

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1")
    ap.add_argument("--temp", type=float, default=0.2)
    args = ap.parse_args()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": "BRIDGE_READY=1"},
        *FEWSHOTS,
    ]

    print("Bridge ready (roots = {} ). Ctrl+C to exit.".format(", ".join(str(r) for r in ALLOWED_ROOTS)))
    while True:
        try:
            user = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        messages.append({"role": "user", "content": user})
        resp = call_ollama(args.model, messages, stream=False,
                           options={"temperature": args.temp})
        content = resp["message"]["content"]
        print(f"\nModel:\n{content}")
        messages.append({"role": "assistant", "content": content})

        kind, match = extract_directive(content)
        if not kind:
            continue

        if kind == "LS":
            out = do_ls(match.group("path"))
        elif kind == "LSR":
            out = do_lsr(match.group("path"), int(match.group("depth")))
        elif kind == "CAT":
            out = do_cat(match.group("path"), match.group("start"), match.group("end"))
        elif kind == "STAT":
            out = do_stat(match.group("path"))
        elif kind == "WRITE":
            out = do_write(match.group("path"), match.group("data"))
        elif kind == "APPEND":
            out = do_append(match.group("path"), match.group("data"))
        elif kind == "DELETE":
            out = do_delete(match.group("path"))
        else:
            out = "ERROR: Unknown directive"

        print(f"\n[Bridge]\n{out}")
        messages.append({"role": "assistant", "content": f"<<BRIDGE>>\n{out}"})

        follow = call_ollama(args.model, messages, stream=False,
                             options={"temperature": args.temp})
        follow_text = follow["message"]["content"]
        print(f"\nModel:\n{follow_text}")
        messages.append({"role": "assistant", "content": follow_text})

if __name__ == "__main__":
    main()
