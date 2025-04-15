import os
import subprocess

def verify_flows_for_all_json(json_dir, script_path):
    """
    For each JSON file in the directory, extract the project name and run the verification script.
    Stream stdout and stderr in real time.
    """
    for filename in os.listdir(json_dir):
        if not filename.endswith('.json'):
            continue

        json_path = os.path.join(json_dir, filename)
        project_name = os.path.splitext(filename)[0]

        print(f"\n=== Verifying: {project_name} ===")

        process = subprocess.Popen(
            ['python', script_path, project_name],
            cwd='backend',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Stream stdout
        for line in process.stdout:
            print(f"[stdout] {line.strip()}")

        # Stream stderr
        for line in process.stderr:
            print(f"[stderr] {line.strip()}")

        process.wait()

        if process.returncode == 0:
            print(f"[✓] Successfully verified flows for {project_name}")
        else:
            print(f"[✗] Failed to verify flows for {project_name} (exit code {process.returncode})")

if __name__ == '__main__':
    json_directory = os.path.join("testing", "Labeling", "FlowData")
    script_path = os.path.join("src", "bert", "inference", "bert_verify_sarif.py")
    verify_flows_for_all_json(json_directory, script_path)
