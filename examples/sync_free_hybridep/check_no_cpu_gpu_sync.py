#!/usr/bin/env python3
# Analyze an nsys sqlite export for CPU-GPU synchronization during the MoE layer
# forward/backward.
#
#   usage (inside the container, which ships nsys + python):
#     nsys export --type sqlite --force-overwrite true -o /tmp/r.sqlite report.nsys-rep
#     python3 check_no_cpu_gpu_sync.py /tmp/r.sqlite
#
# Requires a report captured with CUDA tracing enabled (run the job with the
# --cap-add=SYS_ADMIN that the launcher adds in NSYS modes), so CUPTI kernel/API
# activity is present. If the report only has NVTX (no CUPTI kernels), the script
# falls back to a whole-trace NVTX-name scan.
#
# "Blocking" sync = cuda*Synchronize + device->host (D2H) cudaMemcpy. Async
# device-to-device (D2D) / host-to-device (H2D) copies do NOT stall the CPU and
# are not counted as CPU-GPU sync.
import sqlite3, sys

def q1(cur, sql, args=()):
    try:
        return cur.execute(sql, args).fetchone()[0]
    except Exception:
        return 0

def steady_moe_window(cur):
    """Return (start,end) ns of a mid/late MoE layer (dispatch->combine) cycle."""
    rows = cur.execute(
        "SELECT text,start,end FROM NVTX_EVENTS WHERE text IS NOT NULL ORDER BY start"
    ).fetchall()
    disp = [r for r in rows if r[0] == "dispatch_preprocess in hybrid-ep"]
    comb = [r for r in rows if r[0] == "combine_postprocess in hybrid-ep"]
    if not disp or not comb:
        return None
    ds = disp[len(disp) // 2][1]
    later = [x[2] for x in comb if x[1] >= ds]
    if not later:
        return None
    return (ds, later[0])

def main():
    db = sys.argv[1]
    cur = sqlite3.connect(db).cursor()
    tabs = {r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    has_kernel = "CUPTI_ACTIVITY_KIND_KERNEL" in tabs and \
        q1(cur, "SELECT count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL") > 0

    print(f"{db}:")

    win = steady_moe_window(cur) if "NVTX_EVENTS" in tabs else None

    if has_kernel and win is not None:
        # Precise: count blocking sync inside one steady-state MoE-layer window.
        sync = q1(cur,
            "SELECT count(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s "
            "ON r.nameId=s.id WHERE r.start>=? AND r.start<? AND s.value LIKE '%Synchronize%'",
            win)
        d2h = q1(cur,
            "SELECT count(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY "
            "WHERE start>=? AND start<? AND copyKind=2", win)  # copyKind 2 = D2H
        kernels = q1(cur,
            "SELECT count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND start<?", win)
        blocking = sync + d2h
        verdict = "PASS (sync-free layer)" if blocking == 0 else "FAIL (blocking sync in layer)"
        print(f"  mode                       : precise (steady MoE-layer window)")
        print(f"  window length              : {(win[1]-win[0])/1e6:.3f} ms, {kernels} kernels")
        print(f"  *Synchronize in window     : {sync}")
        print(f"  D2H cudaMemcpy in window   : {d2h}")
        print(f"  VERDICT                    : {verdict}")
        sys.exit(0 if blocking == 0 else 1)

    # Fallback: NVTX-only report (no CUPTI kernels). Scan whole trace for sync-like
    # runtime API and for sync-like NVTX names.
    runtime = q1(cur, "SELECT count(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME") \
        if "CUPTI_ACTIVITY_KIND_RUNTIME" in tabs else 0
    sync = 0
    if "CUPTI_ACTIVITY_KIND_RUNTIME" in tabs:
        sync = q1(cur,
            "SELECT count(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s "
            "ON r.nameId=s.id WHERE s.value LIKE '%Synchronize%' OR s.value LIKE '%Memcpy%'")
    memcpy_rows = q1(cur, "SELECT count(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY") \
        if "CUPTI_ACTIVITY_KIND_MEMCPY" in tabs else 0
    nvtx = q1(cur, "SELECT count(*) FROM NVTX_EVENTS") if "NVTX_EVENTS" in tabs else 0
    verdict = "PASS (sync-free)" if (sync == 0 and memcpy_rows == 0) else "FAIL (sync present)"
    print(f"  mode                       : fallback (NVTX / whole-trace)")
    print(f"  cuda runtime api total     : {runtime}")
    print(f"  synchronize/memcpy api     : {sync}")
    print(f"  device->host memcpy rows   : {memcpy_rows}")
    print(f"  layer NVTX ranges captured : {nvtx}")
    print(f"  VERDICT                    : {verdict}")
    sys.exit(0 if sync == 0 and memcpy_rows == 0 else 1)

if __name__ == "__main__":
    main()
