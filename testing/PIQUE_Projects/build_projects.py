"""
This script takes in the projects from testing/Advisory/top_advisories.txt
and attempts to create a CodeQL database for each project.
This is done by attempting to create a database for each project with 
Java 21, 17, 11, and 8.
If any of these attempts results in a status code of 0, the project is considered
to be buildable with that version of Java.

The metadata for each project is stored in a dictionary, containing 
repoName, projectVersion, javaVersion, projectName, and url.
The output is a JSON file containing the metadata for each project.
Along with a xlsx file.
"""

import os
import requests
import subprocess
import zipfile
import json
import sys
import re
from openpyxl import Workbook
from openpyxl.styles import PatternFill
import concurrent.futures


input_projects = os.path.join("testing", "Advisory", "clean_advisories.txt")
java_versions = ["21", "8", "17", "11", ]
DOWNLOAD_DIR = os.path.join("testing", "PIQUE_Projects", "project_downloads")
DATABASE_DIR = os.path.join("testing", "PIQUE_Projects", "databases")
OUTPUT_DIR = os.path.join("testing", "PIQUE_Projects", "output")


def read_projects(file_path):
    """Reads the project list from a file."""
    with open(file_path, 'r') as f:
        projects = f.readlines()
    projects = [line.strip() for line in projects if line.strip()]
    return projects


def download_projects(projects):
    """
    Downloads the latest source zip archive for each GitHub project from a list.
    Extracts the zip, deletes it, and records metadata for each project.
    """
    meta_data = []

    for project in projects:
        owner, repo = project.split('/')

        # Get the latest release info from GitHub API
        api_url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
        response = requests.get(api_url)

        if response.status_code != 200:
            print(f"Failed to fetch release info for {project}")
            continue

        release_data = response.json()
        tag_name = release_data.get('tag_name')

        if not tag_name:
            print(f"No tag name found for {project}")
            continue

        # Construct the source zip URL for the release tag
        zip_url = f"https://github.com/{owner}/{repo}/archive/refs/tags/{tag_name}.zip"
        zip_name = f"{repo}-{tag_name}.zip"
        zip_path = os.path.join(DOWNLOAD_DIR, zip_name)

        print(f"Downloading {zip_name} from {zip_url}")

        try:
            zip_response = requests.get(zip_url)
            zip_response.raise_for_status()

            # Save the zip file to disk
            with open(zip_path, 'wb') as f:
                f.write(zip_response.content)

            print(f"Saved {zip_name} to {zip_path}")

            # Extract the zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(DOWNLOAD_DIR)

            # Remove the zip after extraction
            os.remove(zip_path)

            # Get the actual extracted folder name
            extracted_dirs = [
                name for name in os.listdir(DOWNLOAD_DIR)
                if os.path.isdir(os.path.join(DOWNLOAD_DIR, name)) and name.startswith(repo)
            ]

            if not extracted_dirs:
                print(f"Could not find extracted directory for {zip_name}")
                continue

            actual_project_name = extracted_dirs[0]
            extract_dir = os.path.join(DOWNLOAD_DIR, actual_project_name)

            # Store metadata
            meta_data.append({
                "repoName": project,
                "projectVersion": tag_name,
                "projectName": actual_project_name,
                "url": zip_url,
                "sourceRoot": extract_dir
            })

        except Exception as e:
            print(f"Failed to process {zip_url}: {e}")

    return meta_data


def change_java_version(java_version):
    """
    Prepares environment to use specified JDK version.
    Forces Java to be first in PATH.
    """
    java_home = os.path.join("C:\\", "Program Files", "Java", f"jdk-{java_version}")
    java_bin = os.path.join(java_home, "bin")
    java_exec = os.path.join(java_bin, "java.exe")

    # Override environment for subprocess
    env = os.environ.copy()
    env["JAVA_HOME"] = java_home
    env["PATH"] = java_bin + os.pathsep + env.get("PATH", "")

    try:
        result = subprocess.run([java_exec, "-version"], env=env, capture_output=True, text=True)
        print(f"Java version set to {java_version}:")
        print(result.stderr.strip())
    except Exception as e:
        print(f"Failed to run Java {java_version}: {e}")

    return env


def create_codeql_database(source_root, db_name, env):
    """
    Attempts to create a CodeQL database for the given source root using the provided environment.
    Also verifies that Java 21 is used by both JAVA_HOME and PATH.
    """
    print(f"\nVerifying Java version for {db_name}...")

    # Check direct JAVA_HOME java.exe
    java_exec = os.path.join(env["JAVA_HOME"], "bin", "java.exe")
    java_check = subprocess.run([java_exec, "-version"], env=env, capture_output=True, text=True)
    print("Explicit java version from JAVA_HOME:")
    print(java_check.stderr.strip())

    # Log first PATH entry
    print("\nPATH being used by CodeQL:")
    print(env["PATH"].split(os.pathsep)[0])

    # Build command
    command = [
        "codeql", "database", "create", db_name,
        "--language=java",
        "--overwrite",
        f"--source-root={source_root}"
    ]
    print(f"\nRunning CodeQL database create for {db_name}...\n")

    # Run command with real-time streaming output
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    java_version_pattern = re.compile(r"Java version:.*", re.IGNORECASE)


    # for line in process.stdout:
    #     sys.stdout.write(line)
    #     sys.stdout.flush()

    for line in process.stdout:
        if java_version_pattern.search(line):
            print(line.strip())  # Only print the matching Java version line

    process.wait()

    if process.returncode != 0:
        print(f"\nFailed to create CodeQL database for {db_name} with Java {env['JAVA_HOME']}")
    else:
        print(f"\nSuccessfully created CodeQL database for {db_name} with Java {env['JAVA_HOME']}")
    return process.returncode == 0

def write_json(data):
    """
    Writes only successfully built project metadata (i.e., not 'DNB') to a JSON file.
    This output file is used by setup_projects.py to give PIQUE all the projects that build successfully.
    Output file: projects_auto.json
    """
    successful_projects = [proj for proj in data if proj["javaVersion"] != "DNB"]

    output = {"projects": successful_projects}
    output_path = os.path.join(OUTPUT_DIR, "projects_auto.json")

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved successful builds to JSON: {output_path}")


def write_xlsx(data):
    wb = Workbook()
    ws = wb.active
    ws.title = "Build Results"

    # Define column headers
    headers = [
        "repoName", "projectVersion", "javaVersion", "projectName", "Would Build",
        "Accuracy (TP / Total)", "Accuracy With Validation", "Download Link"
    ]
    ws.append(headers)

    # Define fill colors
    green_fill = PatternFill(start_color="bfd6ac", end_color="bfd6ac", fill_type="solid")  # light green
    red_fill = PatternFill(start_color="d69c9b", end_color="d69c9b", fill_type="solid")    # light red

    for project in data:
        would_build = "No" if project["javaVersion"] == "DNB" else "Yes"
        row = [
            project["repoName"],
            project["projectVersion"],
            project["javaVersion"],
            project["projectName"],
            would_build,
            "",  # Accuracy (TP / Total)
            "",  # Accuracy With Validation
            project["url"]
        ]
        ws.append(row)

        # Apply color fill
        fill = red_fill if would_build == "No" else green_fill
        for cell in ws[ws.max_row]:
            cell.fill = fill

    # Save file
    output_file = os.path.join(OUTPUT_DIR, "projects_auto.xlsx")
    wb.save(output_file)
    print(f"Saved XLSX to {output_file}")


def build_project(project):
    source_root = project["sourceRoot"]
    curr_project_metadata = {
        "repoName": project["repoName"],
        "projectVersion": project["projectVersion"],
        "javaVersion": "DNB",
        "projectName": project["projectName"],
        "url": project["url"]
    }

    for java_version in java_versions:
        env = change_java_version(java_version)
        db_output = os.path.join(DATABASE_DIR, f"{project['projectName']}_db")

        if create_codeql_database(source_root, db_output, env):
            curr_project_metadata["javaVersion"] = java_version
            print(f"Successfully built {project['projectName']} with Java {java_version}")
            return curr_project_metadata

    print(f"Failed to build {project['projectName']} with any Java version.")
    return curr_project_metadata


def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(DATABASE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    projects = read_projects(input_projects)
    meta_data = download_projects(projects)

    project_results = []

    # Use a process pool to build different projects concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(build_project, project) for project in meta_data]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            project_results.append(result)

    write_json(project_results)
    write_xlsx(project_results)


if __name__ == "__main__":
    main()
