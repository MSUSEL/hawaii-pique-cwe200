import os
import zipfile
import subprocess
import json
import backslicing

# Directory paths
db_dir = os.path.join("testing", "Dataflow", "db_dir")
query_path = os.path.join("codeql", "codeql-custom-queries-java", "testing", "Backall.ql")
query_results_dir = os.path.join("testing", "Dataflow", "query_results")
processed_json_dir = "processed_jsons"
labeled_dataset_path = os.path.join("testing", "Merge_JSONs", "Labeled_JSONs", "Labeled_dataset.json")

# Ensure the query_results and processed_json directories exist
os.makedirs(query_results_dir, exist_ok=True)
os.makedirs(processed_json_dir, exist_ok=True)

# Load the labeled dataset
with open(labeled_dataset_path, 'r') as labeled_file:
    labeled_dataset = json.load(labeled_file)

# Convert labeled dataset to a dictionary for faster lookup
labeled_dict = {entry["fileName"]: {var["name"] for var in entry["variables"]} for entry in labeled_dataset}

s = "AbstractQueryProtocolModelTest.java", "Analysis.java",

# List of filenames
filenames = [
    "BasicSSHUserPrivateKey.java", "DetectPostBuildStepDescriptor.java", "EasAutoDiscover.java",
    "AWSCodeDeployPublisher.java",  "AbstractSolrMetadataExtractor.java",
    "AdvancedBluetoothDetailsHeaderController.java",  "ArtifactoryChoiceListProvider.java",
    "Assistant.java", "AttachmentProvider.java", "BaseUserController.java", "BondStateMachine.java",
    "CLICommand.java", "ContainerExecDecorator.java", "DefaultResetPasswordRequestResponse.java",
    "DirectoryBrowserSupport.java", "FileDownloader.java", "GithubConfig.java", "GitHubServerConfig.java",
    "GitHubTokenCredentialsCreator.java", "GnssNetworkConnectivityHandler.java", "HttpMethod.java",
    "LiveTableResultsTest.java", "LiveTableResultsTest_#2.java", "MySQLBackupProcessor.java",
    "OHttpSessionManager.java", "OpenstackCredentials.java", "PhoneSubInfoController.java", "Plugin.java",
    "RequestIgnoreBatteryOptimizations.java", "ResetPasswordIT.java", "TemporaryFolder.java",
    "TestlabNotifier.java", "TinfoilScanRecorder.java", "UnsafeAccess.java", "ViewOptionHandler.java",
    "WifiEnterpriseConfig.java", "WifiNetworkDetailsFragment.java"
]

# Dictionary to hold all filtered and processed results
all_filtered_results = []

# Function to unzip a file and remove the zip
def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_path)

# Iterate over each filename
for filename in filenames:
    base_name = filename.replace(".java", "")
    db_path = os.path.join(db_dir, base_name)
    print(f"Processing {base_name}")

    
    # Check if a db exists
    zip_path = f"{db_path}.zip"
    if os.path.exists(zip_path):
        # Unzip and delete the zip file
        unzip_file(zip_path, db_dir)
    elif not os.path.exists(db_path):
        print(f"Database for {base_name} not found.")
        continue  # Skip if no db exists
    
    # Run the CodeQL query
    output_sarif = os.path.join(query_results_dir, f"{base_name}.sarif")
    query_cmd = [
        "codeql", "database", "analyze", db_path, query_path,
        "--format=sarif-latest", "--output", output_sarif,
        "--max-paths=100", "--sarif-add-snippets=true", "--no-group-results", "--threads=12"
    ]
    
    try:
        subprocess.run(query_cmd, check=True)
        print(f"Query run successfully for {base_name}, results saved to {output_sarif}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to run query on {base_name}: {e}")
        continue

    # Pass the SARIF result to backslicing.py and get the JSON result
    backslicing_json = backslicing.run(output_sarif)
    filtered_results = [result for result in backslicing_json if result['fileName'] == filename]

    
    # Process each result to determine if it's sensitive
    processed_results = {
        "fileName": filename,
        "variables": []
    }

    for result in filtered_results:
        for variable in result["variables"]:
            variable_name = variable["name"]
            is_sensitive = "yes" if variable_name in labeled_dict.get(filename, set()) else "no"
            variable["isSensitive"] = is_sensitive
            processed_variable = {
                "name": variable_name,
                "isSensitive": is_sensitive,
                "graph": variable["graph"]
        
        }
        processed_results["variables"].append(processed_variable)
    
    all_filtered_results.append(processed_results)
    print(f"Filtered and processed results saved for {filename}")

# Final output JSON file containing all processed results
final_output = os.path.join(processed_json_dir, "CVE.json")

# Save the final dictionary as a single JSON file
with open(final_output, 'w') as file:
    json.dump(all_filtered_results, file, indent=4)

print(f"All processed results combined and saved to {final_output}")
