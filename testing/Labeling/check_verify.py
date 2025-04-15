import os
import shutil
import subprocess

def verify_flows_for_all_json(json_dir, script_path, files_dir):
    """
    For each JSON file in the directory:
    - Ensure backend/Files/{project_name} exists (create if missing)
    - Copy JSON into it as flowMapsByCWE.json
    - Run the verification script with live stdout/stderr
    """
    for filename in os.listdir(json_dir):
        if not filename.endswith('.json'):
            continue

        json_path = os.path.join(json_dir, filename)
        project_name = os.path.splitext(filename)[0]
        project_dir = os.path.join(files_dir, project_name)
        target_json_path = os.path.join(project_dir, 'flowMapsByCWE.json')

        # Ensure project directory exists
        if not os.path.isdir(project_dir):
            try:
                os.makedirs(project_dir)
                print(f"[+] Created missing project directory: {project_dir}")
            except Exception as e:
                print(f"[?] Failed to create directory for {project_name}: {e}")
                continue

        # Copy the JSON file
        try:
            shutil.copyfile(json_path, target_json_path)
            print(f"[?] Copied {filename} ? {target_json_path}")
        except Exception as e:
            print(f"[?] Failed to copy {filename} ? {target_json_path}")
            print(f"    Reason: {e}")
            continue

        # Run the verification script
        print(f"\n=== Verifying: {project_name} ===")
        process = subprocess.Popen(
            ['python', script_path, project_name],
            cwd='backend',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        for line in process.stdout:
            print(f"[stdout] {line.strip()}")
        for line in process.stderr:
            print(f"[stderr] {line.strip()}")

        process.wait()

        if process.returncode == 0:
            print(f"[?] Successfully verified flows for {project_name}")
        else:
            print(f"[?] Failed to verify flows for {project_name} (exit code {process.returncode})")

if __name__ == '__main__':
    json_directory = 'testing/Labeling/FlowData'
    script_path = 'src/bert/inference/bert_verify_sarif.py'
    files_directory = 'backend/Files'
    verify_flows_for_all_json(json_directory, script_path, files_directory)
