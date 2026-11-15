"""Render a trace JSONL as compact, anonymised one-line-per-event records.

Usage: python3 -I compact_trace.py FILE [--driver engine] [--episode NAME] [--only-blocked]
Fields: step, scenario, tool, blocked(is_blocked_tool), executed, response[:70],
counters (B, K_max, policy total, policy consecutive for this tool), env hash before/after (first 12 hex),
state_changed. Blocked paths are printed once per episode header.
"""
import json, sys, argparse

ap = argparse.ArgumentParser()
ap.add_argument("file")
ap.add_argument("--driver")
ap.add_argument("--episode")
ap.add_argument("--only-blocked", action="store_true")
ap.add_argument("--width", type=int, default=70)
a = ap.parse_args()

def short(h):
    return (h or "")[:12]

cur = None
for line in open(a.file):
    d = json.loads(line)
    if d.get("summary") or d.get("record") == "header":
        continue
    if "tool" not in d:
        continue
    if a.driver and d.get("driver") != a.driver:
        continue
    if a.episode and d.get("episode") != a.episode:
        continue
    if a.only_blocked and not d.get("is_blocked_tool"):
        continue
    ep = (d.get("episode"), d.get("driver"))
    if ep != cur:
        cur = ep
        hdr = {"episode": d.get("episode"), "scenario": d.get("scenario"),
               "blocked_paths": sorted(d.get("blocked_paths") or [])}
        if "protected_tools" in d:
            hdr["fallback_paths(protected)"] = sorted(d.get("protected_tools") or [])
        if d.get("driver"):
            hdr["driver"] = d["driver"]
        print("# " + json.dumps(hdr))
    c = d.get("counters_after") or d.get("counters") or {}
    tool = d["tool"]
    cons = (c.get("policy_consecutive_failures") or {}).get(tool, 0)
    resp = d.get("response")
    if not isinstance(resp, str):
        resp = json.dumps(resp)
    resp = resp.replace("\n", " ")
    if len(resp) > a.width:
        resp = resp[: a.width] + "..."
    rec = {
        "step": d.get("step"),
        "scen": d.get("scenario"),
        "tool": tool,
        "blocked": d.get("is_blocked_tool"),
        "executed": d.get("executed"),
        "resp": resp,
        "B": c.get("B"), "K": c.get("K_max"),
        "pol_total": c.get("policy_total_injections"),
        "pol_consec": cons,
        "h_before": short(d.get("env_state_hash_before")),
        "h_after": short(d.get("env_state_hash_after")),
    }
    print(json.dumps(rec, ensure_ascii=False))
