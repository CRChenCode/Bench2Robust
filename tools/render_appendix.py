"""Render trace JSONL files as anonymised fixed-width tables plus a per-trace summary row.

Usage: python3 -I render_appendix.py SPEC.json > out.md
SPEC: list of {"id","title","file","driver"(opt),"episode"(opt),"path_label","revision"}
Columns: step, tool, blk (tool in blocked set), exec (tau2 tool function actually ran, from the
harness's toolkit instrumentation), response (first 46 chars), B/K, tot (policy total stochastic
injections after the call), con (policy consecutive-failure count for this tool after the call),
env-state hash before -> after (first 8 hex of sha256 over the tau2 DB dump).
"""
import hashlib, json, sys, os

spec = json.load(open(sys.argv[1]))
base = os.path.dirname(os.path.abspath(sys.argv[1]))

def rows(item):
    out = []
    for line in open(os.path.join(base, item["file"])):
        d = json.loads(line)
        if d.get("summary") or d.get("record") == "header" or "tool" not in d:
            continue
        if item.get("driver") and d.get("driver") != item["driver"]:
            continue
        if item.get("episode") and d.get("episode") != item["episode"]:
            continue
        out.append(d)
    return out

summary = []
blocks = []
for it in spec:
    R = rows(it)
    d0 = R[0]
    c0 = d0.get("counters_after") or d0.get("counters")
    blocked = sorted(d0.get("blocked_paths") or [])
    lines = []
    hdr = f"{'st':>2} {'tool':<31} {'blk':<3} {'exec':<4} {'response':<48} {'B/K':<3} {'tot':>3} {'con':>3}  env hash before->after"
    lines.append(hdr)
    nb = nb_data = nb_exec = nb_state = 0
    max_run = run = 0; last_tool = None
    B = c0.get("B"); K = c0.get("K_max")
    b_reached_then_blocked = False
    tot_seen = 0
    for d in R:
        c = d.get("counters_after") or d.get("counters") or {}
        tool = d["tool"]
        resp = d.get("response")
        resp = resp if isinstance(resp, str) else json.dumps(resp)
        resp = " ".join(resp.split())
        is_err = resp.startswith("Error")
        r = resp[:46] + (".." if len(resp) > 46 else "")
        blk = bool(d.get("is_blocked_tool"))
        tot = c.get("policy_total_injections")
        con = (c.get("policy_consecutive_failures") or {}).get(tool, 0)
        h0 = (d.get("env_state_hash_before") or "")[:8]
        h1 = (d.get("env_state_hash_after") or "")[:8]
        if blk:
            nb += 1
            nb_data += (not is_err)
            nb_exec += bool(d.get("executed"))
            nb_state += (h0 != h1)
            if tot_seen >= (B or 0) and tot_seen > 0:
                b_reached_then_blocked = True
            run = run + 1 if tool == last_tool else 1
            max_run = max(max_run, run)
        else:
            run = 0
        last_tool = tool if blk else None
        tot_seen = tot if tot is not None else tot_seen
        lines.append(f"{d.get('step'):>2} {tool:<31} {'yes' if blk else 'no':<3} {'yes' if d.get('executed') else 'no':<4} {r:<48} {str(B)+'/'+str(K):<3} {tot!s:>3} {con!s:>3}  {h0}->{h1}")
    sha = hashlib.sha256(open(os.path.join(base, it["file"]), "rb").read()).hexdigest()
    summary.append((it["id"], it["path_label"], d0.get("scenario"), nb, max_run, nb_data, nb_exec, nb_state,
                    "yes" if b_reached_then_blocked else "no"))
    blocks.append((it, blocked, lines, sha, d0))

print("| Trace | Path | Scenario | Blocked calls | Max consecutive blocked calls, same tool | Blocked calls that returned data (lapse) | Blocked calls executed | Blocked calls that changed env state | Blocked call made after B was used up by a stochastic injection |")
print("|---|---|---|---|---|---|---|---|---|")
for s in summary:
    print("| " + " | ".join(str(x) for x in s) + " |")
print()
for it, blocked, lines, sha, d0 in blocks:
    print(f"#### {it['id']}. {it['title']}")
    print()
    sc = d0.get("scenario")
    print(f"Path: {it['path_label']}. Revision: {it['revision']}. Scenario label: `{sc}`. "
          f"Blocked paths ({len(blocked)}): `{', '.join(blocked)}`. Trace file sha256 `{sha[:16]}...`.")
    print()
    print("```text")
    for l in lines:
        print(l)
    print("```")
    print()
