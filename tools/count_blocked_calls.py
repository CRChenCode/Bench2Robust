#!/usr/bin/env python3
# Copyright (c) 2026 Anonymous Authors
# SPDX-License-Identifier: CC-BY-NC-4.0
"""Recount the blocked calls in the distributed traces (docs/scenario_control_evidence.md, Sec. 3.2).

For every trace listed in tools/appendix_spec.json (B1-B4, C1-C4) this script counts
the calls to tools in the episode's blocked set and, among them, how many
  - returned data (the agent-visible response is not an explicit error message),
  - were executed (the benchmark tool function actually ran),
  - changed the environment-state hash.
It then prints the totals for paths (b) and (c) and their S2/S3 breakdown, and exits non-zero if any
blocked call on paths (b)-(c) returned data. Standard library only; reads files, writes nothing.

Usage:  python3 -I tools/count_blocked_calls.py <repository root>
"""
import json
import os
import sys


def returned_data(record):
    resp = record.get("response")
    text = resp if isinstance(resp, str) else json.dumps(resp)
    head = text.lstrip()
    return not (head.startswith("Error") or head.startswith('{"error"') or '"error"' in head[:40])


def main(root):
    spec_path = os.path.join(root, "tools", "appendix_spec.json")
    with open(spec_path, encoding="utf-8") as fh:
        spec = json.load(fh)
    rows = {}
    print(f"{'trace':5} {'path':26} {'scen':4} {'blocked':>7} {'data':>5} {'exec':>5} {'state':>5}  file")
    for entry in spec:
        path = os.path.normpath(os.path.join(os.path.dirname(spec_path), entry["file"]))
        with open(path, encoding="utf-8") as fh:
            recs = [json.loads(line) for line in fh if line.strip()]
        events = [r for r in recs if r.get("record", "event") == "event"]
        if "driver" in entry:
            events = [r for r in events
                      if r.get("driver") == entry["driver"] and r.get("episode") == entry["episode"]]
        blocked = [r for r in events if r.get("is_blocked_tool")]
        labels = {str(r.get("scenario_label") or r.get("scenario")) for r in blocked}
        scen = "S3" if labels & {"S3", "impossible"} else "S2" if labels & {"S2", "switch_needed"} else "?"
        row = dict(n=len(blocked), data=sum(map(returned_data, blocked)),
                   exe=sum(bool(r.get("executed")) for r in blocked),
                   state=sum(bool(r.get("state_changed")) for r in blocked), scen=scen)
        rows[entry["id"]] = row
        print(f"{entry['id']:5} {entry['path_label']:26} {scen:4} {row['n']:7d} {row['data']:5d} "
              f"{row['exe']:5d} {row['state']:5d}  {os.path.relpath(path, root)}")

    groups = {
        "(b) original evaluation harness (B1-B4)": ["B1", "B2", "B3", "B4"],
        "(c) original training environment (C1-C4)": ["C1", "C2", "C3", "C4"],
    }
    print()
    for name, ids in groups.items():
        n = sum(rows[i]["n"] for i in ids)
        d = sum(rows[i]["data"] for i in ids)
        s2 = sum(rows[i]["n"] for i in ids if rows[i]["scen"] == "S2")
        s3 = sum(rows[i]["n"] for i in ids if rows[i]["scen"] == "S3")
        print(f"{name:46} blocked={n:3d} (S2 {s2:3d}, S3 {s3:3d})  returned data={d}")
    bc = [i for ids in groups.values() for i in ids]
    n = sum(rows[i]["n"] for i in bc)
    d = sum(rows[i]["data"] for i in bc)
    s2 = sum(rows[i]["n"] for i in bc if rows[i]["scen"] == "S2")
    s3 = sum(rows[i]["n"] for i in bc if rows[i]["scen"] == "S3")
    print(f"\nPaths (b)-(c): {n} blocked calls (S2 {s2}, S3 {s3}); {d} returned data.")
    return 0 if d == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
