import os
import zipfile
import subprocess
import json

# Directory paths
db_dir = os.path.join("testing", "Dataflow", "db_dir")
query_path = os.path.join("codeql", "codeql-custom-queries-java", "testing", "Backall.ql")
query_results_dir = os.path.join("testing", "Dataflow", "query_results")
processed_json_dir = "processed_jsons"

# Ensure the query_results and processed_json directories exist
os.makedirs(query_results_dir, exist_ok=True)
os.makedirs(processed_json_dir, exist_ok=True)

# List of filenames
filenames = [
    "BasicSSHUserPrivateKey.java", "DetectPostBuildStepDescriptor.java", "EasAutoDiscover.java",
    "AWSCodeDeployPublisher.java", "AbstractQueryProtocolModelTest.java", "AbstractSolrMetadataExtractor.java",
    "AdvancedBluetoothDetailsHeaderController.java", "Analysis.java", "ArtifactoryChoiceListProvider.java",
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

# Function to unzip a file and remove the zip
def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_path)

# Iterate over each filename
for filename in filenames:
    base_name = filename.replace(".java", "")
    db_path = os.path.join(db_dir, base_name)
    
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
        "--max-paths=100", "--sarif-add-snippets=true", "--no-group-results"
    ]
    
    try:
        subprocess.run(query_cmd, check=True)
        print(f"Query run successfully for {base_name}, results saved to {output_sarif}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to run query on {base_name}: {e}")
        continue

    # Pass the SARIF result to backslicing.py
    backslicing_cmd = [
        "python3", "backslicing.py", "--input", output_sarif
    ]
    
    try:
        subprocess.run(backslicing_cmd, check=True)
        print(f"Backslicing completed for {base_name}.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to run backslicing on {base_name}: {e}")

    # Process the resulting JSON from backslicing.py
    backslicing_json = os.path.join(processed_json_dir, f"{base_name}_results.json")
    
    if os.path.exists(backslicing_json):
        with open(backslicing_json, 'r') as file:
            data = json.load(file)
        
        # Extract only the results for the corresponding filename (including .java)
        filtered_results = [result for result in data if result['filename'] == filename]
        
        # Save the filtered results into a new JSON file
        filtered_json_path = os.path.join(processed_json_dir, f"{base_name}_filtered.json")
        with open(filtered_json_path, 'w') as file:
            json.dump(filtered_results, file, indent=4)
        
        print(f"Filtered results saved for {filename} to {filtered_json_path}")

# Final output JSON file containing all filtered results
final_output = os.path.join(processed_json_dir, "final_results.json")
all_filtered_results = []

for filename in filenames:
    base_name = filename.replace(".java", "")
    filtered_json_path = os.path.join(processed_json_dir, f"{base_name}_filtered.json")
    
    if os.path.exists(filtered_json_path):
        with open(filtered_json_path, 'r') as file:
            filtered_results = json.load(file)
            all_filtered_results.extend(filtered_results)

# Save the final combined JSON file
with open(final_output, 'w') as file:
    json.dump(all_filtered_results, file, indent=4)

print(f"All filtered results combined and saved to {final_output}")
