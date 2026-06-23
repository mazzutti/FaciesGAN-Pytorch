import subprocess
from pathlib import Path

cmd_runner = [
    "/home/mazzutti/POSDOC/FaciesGAN/.venv/bin/python", "-u", "-m", "experiments.runner",
    "--input-path", "data",
    "--skip-training",
    "--output-path", "outputs_old/experiments",
    "--model-paths",
    "outputs_old/experiments/wells_seismic",
    "outputs_old/experiments/wells_only",
    "outputs_old/experiments/seismic_only",
    "outputs_old/experiments/unconditional",
    "--uncertainty-index", "100",
    "--uncertainty-samples", "1000",
    "--no-embeddings"
]

print("Running runner command:", " ".join(cmd_runner))
res_runner = subprocess.run(cmd_runner, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

res_report = None
if res_runner.returncode == 0:
    cmd_report = [
        "/home/mazzutti/POSDOC/FaciesGAN/.venv/bin/python", "generate_report.py",
        "--outputs-dir", "outputs_old/experiments",
        "--output", "index.html"
    ]
    print("Running report command:", " ".join(cmd_report))
    res_report = subprocess.run(cmd_report, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
else:
    print("Runner command failed. Skipping report generation.")

log_path = Path("runner_debug.log")
with log_path.open("w", encoding="utf-8") as f:
    f.write("=== RUNNER RETURN CODE ===\n")
    f.write(str(res_runner.returncode) + "\n\n")
    f.write("=== RUNNER STDOUT ===\n")
    f.write(res_runner.stdout + "\n\n")
    f.write("=== RUNNER STDERR ===\n")
    f.write(res_runner.stderr + "\n\n")
    
    if res_report is not None:
        f.write("=== REPORT RETURN CODE ===\n")
        f.write(str(res_report.returncode) + "\n\n")
        f.write("=== REPORT STDOUT ===\n")
        f.write(res_report.stdout + "\n\n")
        f.write("=== REPORT STDERR ===\n")
        f.write(res_report.stderr + "\n")

print(f"Done. Log written to {log_path.resolve()}")
