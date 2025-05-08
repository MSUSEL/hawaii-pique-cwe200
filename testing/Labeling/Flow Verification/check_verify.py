import os
import shutil
import subprocess

"""
This script is used to set up the environment for verifying flow maps in the backend.
It does the following:
1. For each JSON file in the specified directory, it checks if the corresponding project directory exists in backend/Files.
2. If the directory does not exist, it creates it.
3. It copies the JSON file into the project directory as flowMapsByCWE.json.
4. It runs the verification script (bert_verify_sarif.py) with the project name as an argument.
5. It captures and prints the output of the verification script to the console.
6. It prints the result of the verification to the console.

TLDR: This script copies each JSON file from FlowData into the corresponding project directory in backend/Files,
This is done to piggyback on the existing verification script (bert_verify_sarif.py) for flow maps.
While allowing us to compare the labeled data with the results of the verification script.
"""

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
    json_directory = os.path.join("testing", "Labeling", "Flow Verification", "FlowData")
    script_path = os.path.join("src", "bert", "inference", "bert_verify_sarif.py")
    files_directory = os.path.join("backend", "Files")
    verify_flows_for_all_json(json_directory, script_path, files_directory)
