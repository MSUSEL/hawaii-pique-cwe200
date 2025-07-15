import os
import json
import requests

"""
This script downloads projects from the projects.json file and saves them in pique/input/projects.
"""

def download_projects(json_file):
    # Create the download directory if it doesn't exist.
    download_dir = os.path.join("pique", "input", "projects")
    os.makedirs(download_dir, exist_ok=True)

    # Load the JSON data.
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    projects = data.get("projects", [])
    for project in projects:
        url = project.get("url")
        project_name = project.get("projectName")
        if not url or not project_name:
            print(f"Skipping project with missing URL or projectName: {project}")
            continue
        if f"{project_name}.zip" in os.listdir(download_dir):
            print(f"Project {project_name} already exists. Skipping download.")
            continue

        # Define the target filename, e.g., downloaded_projects/cxf-4.1.1.zip
        target_file = os.path.join(download_dir, f"{project_name}.zip")
        try:
            response = requests.get(url, stream=True)
            if response.status_code != 200:
                print(f"Failed to download {url}. HTTP status code: {response.status_code}")
                continue
            
            # Write the contents in chunks.
            with open(target_file, 'wb') as out_file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive new chunks
                        out_file.write(chunk)
            print(f"Successfully downloaded {url}")
        except Exception as e:
            print(f"An error occurred while downloading {url}: {e}")

if __name__ == "__main__":
    # Assuming the JSON file is called "projects.json" and is in the same directory.
    download_projects(os.path.join("testing", "PIQUE_Projects", "projects.json"))
