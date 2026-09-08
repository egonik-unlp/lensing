#!/usr/bin/env python3
"""Instantiation guard — the machine-checkable half of the bootstrap contract.

Three failure modes keep recurring when this template is instantiated, and
prose in CLAUDE.md has not been enough to stop any of them:

  1. Bootstrap is left half-done, and the session improvises on top of a
     placeholder domain.
  2. The "new project" is not a self-contained copy — it is the framework
     checkout itself, or a partial tree missing framework files.
  3. Explorations are run ad hoc instead of through the experiment-designer /
     experiment-runner agents, so runs land with no report and the empirical
     record never grows.

This module turns all three into invariants a hook can enforce. It is the
single source of truth for "is this instance bootstrapped", "is this a real
instance at all", and "is this run launch sanctioned".

Subcommands
-----------
  check [--json]                 verdict on this checkout (exit 0 = ready)
  bootstrap start                mark a bootstrap in flight (Stop is gated)
  bootstrap abort --reason R     release the Stop gate deliberately
  bootstrap status               current bootstrap state
  ticket issue --kind campaign --design D --report P [--allowance N]
  ticket issue --kind oneshot --reason R
  ticket status                  the live run ticket, if any
  ticket close                   verify the record was extended, then clear
  hook <EventName>               Claude Code hook entry point (stdin JSON)

Exit codes for `check`: 0 ready, 1 not bootstrapped, 2 not a valid instance.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Written by tools/package.py into every packaged instance; absent in the
# framework checkout. Presence = "this tree was instantiated from a package".
MANIFEST = ROOT / ".lensing-upstream.json"
# Checked in at the framework repo root and NEVER packaged (not in package.py's
# INCLUDE list), so it marks the mother checkout and only the mother checkout.
MOTHER_MARKER = ROOT / ".lensing-mother"

STATE_DIR = ROOT / ".lensing"
BOOTSTRAP_STATE = STATE_DIR / "bootstrap-state.json"
RUN_TICKET = STATE_DIR / "run-ticket.json"
TICKET_LOG = STATE_DIR / "tickets.log"

# Mirrors render.py's sentinel; duplicated rather than imported so the hook
# path stays dependency-free even when agents-src/ is mid-edit.
TEMPLATE_SENTINEL = "<!-- LENSING-TEMPLATE: bootstrap pending -->"

# Values the shipped placeholder domain.toml carries. A real domain cannot
# match these, so they are a safe "still unconfigured" tell.
PLACEHOLDER_PROJECT_NAME = "unconfigured"
PLACEHOLDER_PROJECT_TITLE = "Unconfigured Lensing Instance"
PLACEHOLDER_COLLECTION = "corpus"

# Paths bootstrap itself legitimately writes before the instance is
# configured. Everything else under the repo root is denied until bootstrap
# completes — that is the whole point of the gate.
BOOTSTRAP_WRITABLE = (
    "domain.toml",
    "CLAUDE.md",
    "PRODUCT.md",
    "pipeline.toml",
    "models.toml",
    "README.md",
    ".env",
    ".gitignore",
    "experiments/",
    "docs/",
    "branding/",
    "agents-src/",
    ".lensing/",
    ".claude/",
    ".agents/",
    ".gemini/",
)

# API mutations that must carry a run ticket. Matched against the whole Bash
# command string. The negative lookahead keeps sub-resources ungated: polling
# GET /api/runs/<id> and POST /api/runs/<id>/stop are normal babysitting.
GATED_ENDPOINTS = (
    ("run launch", re.compile(r"/api/runs(?![/\w-])")),
    ("dataset build", re.compile(r"/api/datasets(?![/\w-])")),
)
POST_RE = re.compile(
    r"(-X\s*POST|--request\s+POST|(?<![\w-])-d[\s@'\"]|--data(-raw|-binary|-urlencode)?[\s=])"
)

WRITE_TOOLS = ("Write", "Edit", "NotebookEdit", "MultiEdit")


# --------------------------------------------------------------------------
# state helpers


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _log(line: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(TICKET_LOG, "a") as f:
        f.write(f"{_now()}  {line}\n")


# --------------------------------------------------------------------------
# provenance: which kind of tree is this?


def mode() -> str:
    """'framework' (the mother checkout), 'instance' (packaged + unpacked),
    or 'unknown' (neither marker — an improvised or partial copy)."""
    if MOTHER_MARKER.exists():
        return "framework"
    if MANIFEST.exists():
        return "instance"
    return "unknown"


def nested_in_mother() -> Path | None:
    """The instance must live in its OWN folder, not inside the framework
    checkout. Returns the offending ancestor if one is a mother repo."""
    for parent in ROOT.parents:
        if (parent / ".lensing-mother").exists():
            return parent
    return None


def missing_framework_files(limit: int = 8) -> tuple[int, list[str]]:
    """Framework files the provenance manifest promises but that are absent —
    i.e. the copy is partial. Rendered outputs (.claude/.agents/.gemini) are
    exempt: render.py legitimately drops some of them (e.g. ingestion=false)."""
    manifest = _read_json(MANIFEST)
    files = manifest.get("files") or {}
    missing = [
        rel
        for rel, meta in sorted(files.items())
        if meta.get("kind") == "framework" and not (ROOT / rel).exists()
    ]
    return len(missing), missing[:limit]


# --------------------------------------------------------------------------
# the verdict


def _domain_findings() -> list[str]:
    """Reasons domain.toml still looks like the shipped placeholder."""
    path = ROOT / "domain.toml"
    if not path.exists():
        return ["domain.toml is missing"]
    try:
        with open(path, "rb") as f:
            d = tomllib.load(f)
    except Exception as e:
        return [f"domain.toml does not parse: {e}"]

    out = []
    project = d.get("project", {})
    if project.get("name") == PLACEHOLDER_PROJECT_NAME:
        out.append('[project] name is still "unconfigured"')
    if project.get("title") == PLACEHOLDER_PROJECT_TITLE:
        out.append("[project] title is still the placeholder")
    if d.get("corpus", {}).get("collection") == PLACEHOLDER_COLLECTION:
        out.append('[corpus] collection is still the placeholder "corpus"')
    examples = [
        f.get("name") for f in d.get("fields", []) if str(f.get("name", "")).endswith("_example")
    ]
    if examples:
        out.append(f"placeholder fields still declared: {', '.join(examples)}")
    return out


def _render_drift() -> str | None:
    """None if the agent layer is in sync with agents-src/ + domain.toml."""
    script = ROOT / "agents-src" / "render.py"
    if not script.exists():
        return "agents-src/render.py is missing"
    try:
        p = subprocess.run(
            [sys.executable, str(script), "--check"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception as e:
        return f"render check could not run: {e}"
    if p.returncode != 0:
        tail = (p.stderr or p.stdout or "").strip().splitlines()
        return tail[-1] if tail else "agent layer is out of sync with agents-src/"
    return None


def verdict() -> dict:
    """Every invariant, with the tree's overall status.

    status: framework | ready | incomplete | invalid
    """
    m = mode()
    checks: list[dict] = []

    def add(key: str, ok: bool, detail: str) -> None:
        checks.append({"key": key, "ok": ok, "detail": detail})

    if m == "framework":
        return {
            "mode": m,
            "status": "framework",
            "checks": [],
            "summary": (
                "This is the lensing FRAMEWORK checkout, not an instance. "
                "Instantiate with `zig build package` and unpack the result "
                "somewhere else; never bootstrap in place here."
            ),
        }

    # --- provenance (rule 2: own folder, full copy) ---
    if m == "unknown":
        add(
            "provenance",
            False,
            "neither .lensing-upstream.json nor .lensing-mother is present — this "
            "tree was not produced by `zig build package`",
        )
    else:
        add("provenance", True, "packaged instance (.lensing-upstream.json present)")

    nest = nested_in_mother()
    add(
        "own-folder",
        nest is None,
        "self-contained folder" if nest is None else f"nested inside the framework checkout at {nest}",
    )

    if m == "instance":
        n_missing, sample = missing_framework_files()
        add(
            "complete-copy",
            n_missing == 0,
            "all framework files present"
            if n_missing == 0
            else f"{n_missing} framework file(s) missing, e.g. {', '.join(sample)}",
        )

    # --- domain (rule 1: bootstrap finishes) ---
    claude_md = ROOT / "CLAUDE.md"
    if not claude_md.exists():
        add("claude-md", False, "CLAUDE.md is missing (render-agents never ran)")
    elif TEMPLATE_SENTINEL in claude_md.read_text():
        add("claude-md", False, "CLAUDE.md is still the bootstrap-pending stub")
    else:
        add("claude-md", True, "CLAUDE.md is rendered for this domain")

    dom = _domain_findings()
    add("domain", not dom, "domain.toml describes a real domain" if not dom else "; ".join(dom))

    drift = _render_drift()
    add("agent-layer", drift is None, "rendered agent layer is in sync" if drift is None else drift)

    # The hooks are what make any of this binding. They live under .claude/,
    # which the manifest classifies as "rendered" and therefore exempts from
    # the completeness check — so check them explicitly, or an instance could
    # quietly lose every gate and still report itself sound.
    settings = ROOT / ".claude" / "settings.json"
    wired = settings.exists() and "lensing_guard.py" in settings.read_text()
    add(
        "guard-hooks",
        wired,
        ".claude/settings.json wires the guard hooks"
        if wired
        else ".claude/settings.json is missing or no longer calls tools/lensing_guard.py — "
        "restore it from the framework, or the invariants are unenforced",
    )

    # --- record (rule 3 needs somewhere to write) ---
    facts = ROOT / "experiments" / "PROJECT-FACTS.md"
    add(
        "record",
        facts.exists(),
        "experiments/PROJECT-FACTS.md present" if facts.exists() else "experiments/PROJECT-FACTS.md is missing",
    )

    failed = [c for c in checks if not c["ok"]]
    hard = {"provenance", "own-folder", "complete-copy"}
    if any(c["key"] in hard for c in failed):
        status = "invalid"
    elif failed:
        status = "incomplete"
    else:
        status = "ready"

    return {
        "mode": m,
        "status": status,
        "checks": checks,
        "summary": f"{len(checks) - len(failed)}/{len(checks)} invariants met",
    }


def bootstrap_state() -> dict:
    return _read_json(BOOTSTRAP_STATE)


# --------------------------------------------------------------------------
# run tickets (rule 3: explorations go through the agents)


def ticket() -> dict:
    return _read_json(RUN_TICKET)


def ticket_issue(kind: str, design: str, report: str, reason: str, allowance: int) -> int:
    v = verdict()
    if kind == "campaign":
        if not design or not report:
            print("ticket: a campaign ticket needs --design and --report", file=sys.stderr)
            return 1
        if not report.startswith("experiments/") or not report.endswith(".md"):
            print(
                f"ticket: --report must name a campaign report under experiments/ (got {report!r})",
                file=sys.stderr,
            )
            return 1
    elif kind == "oneshot":
        if not reason:
            print(
                "ticket: a one-shot ticket needs --reason quoting what the user asked for",
                file=sys.stderr,
            )
            return 1
        allowance = 1
    else:
        print(f"ticket: unknown kind {kind!r}", file=sys.stderr)
        return 1

    if v["status"] not in ("ready", "framework"):
        print(
            f"ticket: refusing to issue — instance is {v['status']} ({v['summary']}). "
            "Finish /bootstrap first.",
            file=sys.stderr,
        )
        return 1

    existing = ticket()
    if existing and existing.get("allowance", 0) > existing.get("used", 0):
        print(
            f"ticket: a {existing.get('kind')} ticket is already open "
            f"({existing.get('used', 0)}/{existing.get('allowance')} used). "
            "Close it before issuing another.",
            file=sys.stderr,
        )
        return 1

    t = {
        "kind": kind,
        "design": design,
        "report": report,
        "reason": reason,
        "allowance": allowance,
        "used": 0,
        "issued_at": _now(),
        "issued_at_epoch": time.time(),
    }
    _write_json(RUN_TICKET, t)
    _log(f"ISSUE kind={kind} allowance={allowance} design={design!r} report={report!r} reason={reason!r}")
    print(f"ticket issued: {kind}, {allowance} gated call(s)" + (f", report {report}" if report else ""))
    return 0


def ticket_consume(what: str) -> tuple[bool, str]:
    """Spend one gated call. Returns (allowed, message)."""
    t = ticket()
    if not t:
        return False, "no run ticket is open"
    used, allowance = t.get("used", 0), t.get("allowance", 0)
    if used >= allowance:
        return False, f"the open {t.get('kind')} ticket is spent ({used}/{allowance})"
    t["used"] = used + 1
    _write_json(RUN_TICKET, t)
    _log(f"SPEND {what} {t['used']}/{allowance} kind={t.get('kind')}")
    return True, f"{t.get('kind')} ticket {t['used']}/{allowance}"


def ticket_close() -> int:
    t = ticket()
    if not t:
        print("ticket: nothing open")
        return 0
    if t.get("kind") == "campaign" and t.get("used", 0) > 0:
        report = ROOT / t.get("report", "")
        if not report.exists():
            print(
                f"ticket: refusing to close — {t.get('report')} does not exist. "
                f"{t.get('used')} run(s) were launched under this ticket; the campaign "
                "report is how the empirical record grows.",
                file=sys.stderr,
            )
            return 1
        facts = ROOT / "experiments" / "PROJECT-FACTS.md"
        if facts.exists() and facts.stat().st_mtime < t.get("issued_at_epoch", 0):
            print(
                "ticket: refusing to close — experiments/PROJECT-FACTS.md has not been "
                "touched since this ticket was issued. Reconcile the leaderboard / "
                "pitfalls / lineage before closing.",
                file=sys.stderr,
            )
            return 1
    RUN_TICKET.unlink(missing_ok=True)
    _log(f"CLOSE kind={t.get('kind')} used={t.get('used')} report={t.get('report')!r}")
    print("ticket closed")
    return 0


def unreconciled_ticket() -> dict | None:
    """An open campaign ticket that spent runs but has no report on disk."""
    t = ticket()
    if not t or t.get("kind") != "campaign" or t.get("used", 0) < 1:
        return None
    if (ROOT / t.get("report", "")).exists():
        return None
    return t


# --------------------------------------------------------------------------
# rendering


def render_verdict(v: dict) -> str:
    lines = [f"lensing instance: {v['status'].upper()} — {v['summary']}"]
    for c in v["checks"]:
        lines.append(f"  {'✓' if c['ok'] else '✗'} {c['key']}: {c['detail']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# hooks


def _deny(reason: str) -> int:
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
        )
    )
    return 0


def _context(event: str, text: str) -> int:
    print(json.dumps({"hookSpecificOutput": {"hookEventName": event, "additionalContext": text}}))
    return 0


def _block_stop(reason: str) -> int:
    print(json.dumps({"decision": "block", "reason": reason}))
    return 0


def _rel_to_root(p: str) -> str | None:
    """Repo-relative posix path, or None when the path is outside the repo
    (scratchpad, /tmp — never gated)."""
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except Exception:
        return None


def hook_session_start(payload: dict) -> int:
    v = verdict()
    if v["status"] == "framework":
        return 0
    if v["status"] == "ready":
        t = ticket()
        if t:
            return _context(
                "SessionStart",
                f"lensing: an open {t.get('kind')} run ticket is carried over "
                f"({t.get('used')}/{t.get('allowance')} used, report {t.get('report') or 'n/a'}). "
                "Finish the campaign report and `ticket close`, or close it before new work.",
            )
        return 0

    st = bootstrap_state().get("status")
    lead = {
        "invalid": (
            "This directory is NOT a valid lensing instance. Do not start project work here. "
            "A new project must be its own folder holding a full copy produced by "
            "`zig build package` (unpack the tarball elsewhere), never the framework "
            "checkout and never a partial tree."
        ),
        "incomplete": (
            "This lensing instance is NOT bootstrapped. `/bootstrap` is the only work that "
            "may proceed here — no datasets, no runs, no exploration, no improvising on the "
            "placeholder domain. Bootstrap always finishes: once started it runs to a "
            "rendered CLAUDE.md and a real domain.toml, or it is explicitly aborted."
        ),
    }[v["status"]]
    extra = ""
    if st == "in_progress":
        extra = "\nA bootstrap is already in flight — resume it at the phase the repo state implies."
    elif st == "aborted":
        extra = f"\nA previous bootstrap was aborted: {bootstrap_state().get('reason', '')!r}"
    return _context("SessionStart", f"{lead}\n\n{render_verdict(v)}{extra}")


def hook_pre_tool_use(payload: dict) -> int:
    v = verdict()
    if v["status"] == "framework":
        return 0

    tool = payload.get("tool_name", "")
    ti = payload.get("tool_input") or {}

    # --- rule 3: run launches / dataset builds need a ticket ---
    if tool == "Bash":
        cmd = str(ti.get("command", ""))
        for what, rx in GATED_ENDPOINTS:
            if rx.search(cmd) and POST_RE.search(cmd):
                if v["status"] != "ready":
                    return _deny(
                        f"Blocked {what}: this instance is {v['status']}.\n{render_verdict(v)}\n"
                        "Finish /bootstrap before touching the API."
                    )
                ok, msg = ticket_consume(what)
                if ok:
                    return 0
                return _deny(
                    f"Blocked {what}: {msg}.\n\n"
                    "Experiments are not launched ad hoc — they go through the "
                    "experiment-designer agent (design, with the user) and then the "
                    "experiment-runner agent (execute, report, reconcile "
                    "experiments/PROJECT-FACTS.md). That is what keeps every run "
                    "attached to a campaign report.\n\n"
                    "The runner opens its ticket with:\n"
                    "  python3 tools/lensing_guard.py ticket issue --kind campaign "
                    "--design <slug> --report experiments/<date>-<slug>.md --allowance <N>\n\n"
                    "Escape hatch — ONLY when the user explicitly asked for a single ad-hoc "
                    "run (never for a scan, sweep or exploration):\n"
                    "  python3 tools/lensing_guard.py ticket issue --kind oneshot "
                    '--reason "<what the user asked for>"'
                )

    if tool not in WRITE_TOOLS:
        return 0

    # Paths outside the repo (scratchpad, /tmp) are never gated.
    rel = _rel_to_root(str(ti.get("file_path") or ti.get("notebook_path") or ""))
    if rel is None:
        return 0

    # The ticket is the audit trail; hand-writing it would defeat the point.
    if rel == ".lensing/run-ticket.json":
        return _deny(
            "The run ticket is issued through `python3 tools/lensing_guard.py ticket issue`, "
            "never written by hand — the CLI is what records the report path and the audit log."
        )

    # --- rules 1 & 2: no project work on an unbootstrapped tree ---
    if v["status"] in ("incomplete", "invalid") and not any(
        rel == a or rel.startswith(a) for a in BOOTSTRAP_WRITABLE
    ):
        return _deny(
            f"Blocked write to {rel}: this instance is {v['status']} and only bootstrap's own "
            f"files may be written until it completes.\n{render_verdict(v)}\n\n"
            "Run /bootstrap. If this write really is part of bootstrap, it belongs under one "
            f"of: {', '.join(BOOTSTRAP_WRITABLE)}"
        )
    return 0


def hook_stop(payload: dict) -> int:
    # stop_hook_active means we already blocked once this turn — never loop.
    if payload.get("stop_hook_active"):
        return 0
    v = verdict()
    if v["status"] == "framework":
        return 0

    # rule 1: bootstrap always finishes.
    if v["status"] in ("incomplete", "invalid"):
        if bootstrap_state().get("status") == "in_progress":
            return _block_stop(
                "Bootstrap is in flight and NOT finished — bootstrap always finishes.\n"
                f"{render_verdict(v)}\n\n"
                "Continue /bootstrap from the phase the repo state implies. If the user told you "
                "to stop, record that explicitly first:\n"
                '  python3 tools/lensing_guard.py bootstrap abort --reason "<why>"'
            )
        return 0

    # rule 3: runs happened, the record must have grown.
    t = unreconciled_ticket()
    if t:
        return _block_stop(
            f"{t.get('used')} run(s) were launched under the campaign ticket "
            f"{t.get('design')!r}, but {t.get('report')} does not exist. A campaign is not "
            "finished until the report is written and experiments/PROJECT-FACTS.md is "
            "reconciled (leaderboard, noise bands, lineage, pitfalls).\n"
            "Write the report, then: python3 tools/lensing_guard.py ticket close"
        )
    return 0


HOOKS = {
    "SessionStart": hook_session_start,
    "PreToolUse": hook_pre_tool_use,
    "Stop": hook_stop,
    "SubagentStop": hook_stop,
}


def hook_main(event: str) -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception:
        payload = {}
    fn = HOOKS.get(event)
    if fn is None:
        return 0
    try:
        return fn(payload)
    except Exception as e:
        # A guard that crashes must never wedge the session; say so and pass.
        print(f"lensing_guard: {event} hook error: {e}", file=sys.stderr)
        return 0


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(prog="lensing_guard")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="verdict on this checkout")
    c.add_argument("--json", action="store_true")

    b = sub.add_parser("bootstrap", help="bootstrap lifecycle state")
    bsub = b.add_subparsers(dest="bcmd", required=True)
    bsub.add_parser("start")
    ab = bsub.add_parser("abort")
    ab.add_argument("--reason", required=True)
    bsub.add_parser("status")

    t = sub.add_parser("ticket", help="run tickets for API mutations")
    tsub = t.add_subparsers(dest="tcmd", required=True)
    ti = tsub.add_parser("issue")
    ti.add_argument("--kind", choices=("campaign", "oneshot"), required=True)
    ti.add_argument("--design", default="")
    ti.add_argument("--report", default="")
    ti.add_argument("--reason", default="")
    ti.add_argument("--allowance", type=int, default=1)
    tsub.add_parser("status")
    tsub.add_parser("close")

    h = sub.add_parser("hook", help="Claude Code hook entry point")
    h.add_argument("event")

    a = ap.parse_args()

    if a.cmd == "check":
        v = verdict()
        if a.json:
            print(json.dumps(v, indent=2))
        else:
            print(render_verdict(v) if v["checks"] else f"lensing: {v['summary']}")
        return {"ready": 0, "framework": 0, "incomplete": 1, "invalid": 2}[v["status"]]

    if a.cmd == "bootstrap":
        if a.bcmd == "start":
            if mode() == "framework":
                print(
                    "bootstrap: this is the lensing FRAMEWORK checkout. A new project gets its "
                    "own folder: `zig build package`, unpack dist/lensing.tar.gz elsewhere, and "
                    "bootstrap there. Refusing to start in place.",
                    file=sys.stderr,
                )
                return 2
            _write_json(BOOTSTRAP_STATE, {"status": "in_progress", "started_at": _now()})
            _log("BOOTSTRAP start")
            print("bootstrap marked in flight — Stop is gated until it finishes or is aborted")
            return 0
        if a.bcmd == "abort":
            _write_json(
                BOOTSTRAP_STATE, {"status": "aborted", "reason": a.reason, "aborted_at": _now()}
            )
            _log(f"BOOTSTRAP abort reason={a.reason!r}")
            print("bootstrap aborted — the Stop gate is released")
            return 0
        print(json.dumps(bootstrap_state() or {"status": "not_started"}, indent=2))
        return 0

    if a.cmd == "ticket":
        if a.tcmd == "issue":
            return ticket_issue(a.kind, a.design, a.report, a.reason, a.allowance)
        if a.tcmd == "close":
            return ticket_close()
        t = ticket()
        print(json.dumps(t, indent=2) if t else "no ticket open")
        return 0

    if a.cmd == "hook":
        return hook_main(a.event)

    return 0


if __name__ == "__main__":
    sys.exit(main())
