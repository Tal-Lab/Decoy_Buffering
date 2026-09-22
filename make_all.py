"""Regenerate every dataset, figure and reported number from scratch.

    python make_all.py            # full pipeline
    python make_all.py --figures  # skip the sweeps, rebuild figures only

Each script is standalone and recomputes what it needs, so they can also be
run individually. Total runtime is dominated by the age-structured
replication sweeps (scripts 04-08).
"""
import os, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
SWEEPS = ["01_sweep_thinning", "02_sweep_extrinsic", "03_scan_amplification",
          "04_sweep_replication", "05_sweep_modulation", "06_decompose_cellcycle",
          "07_scan_cancellation", "08_control_beta0",
          "12_scan_iteron", "13_export_age_profiles",
          "14_scan_iteron_biological"]
FIGURES = ["09_figure1", "10_figure2", "11_figure3", "15_figureS1"]

steps = FIGURES if "--figures" in sys.argv else SWEEPS + FIGURES
failed = []
for name in steps:
    t0 = time.time()
    print(f"[{name}] running ...", flush=True)
    r = subprocess.run([sys.executable, os.path.join("scripts", name + ".py")],
                       capture_output=True, text=True)
    status = "ok" if r.returncode == 0 else "FAILED"
    print(f"[{name}] {status} in {time.time()-t0:.0f}s", flush=True)
    if r.returncode:
        failed.append(name)
        print(r.stderr[-1500:], flush=True)

for extra in ("reported_values.py", "make_s1_data.py", "verify.py"):
    if "--figures" in sys.argv and extra != "verify.py":
        continue
    t0 = time.time()
    print(f"[{extra}] running ...", flush=True)
    r = subprocess.run([sys.executable, extra], capture_output=True, text=True)
    print(r.stdout[-2000:], flush=True)
    if r.returncode:
        failed.append(extra)
        print(r.stderr[-1000:], flush=True)
    print(f"[{extra}] {'ok' if not r.returncode else 'FAILED'} in {time.time()-t0:.0f}s", flush=True)

print("\nALL STEPS OK" if not failed else f"\nFAILED: {failed}")
sys.exit(1 if failed else 0)
