# CWE-200 Web App Tool Repo #

### Quick summary ###

This tool is designed to detect and identify vulnerabilities related to CWE-200 (Exposure of Sensitive Information to an Unauthorized Actor) within Java projects. 
Sensitive information may be inadvertently exposed through various methods, including logging activities, revealing underlying program logic, displaying directory information, 
or printing usernames and passwords, among others. Given the complexity and subtlety of these exposures, our tool offers a comprehensive scanning solution. 
Users can upload their Java projects for a thorough analysis, ensuring all instances of such vulnerabilities are identified.

### Current Coverage ###

- CWE-200: Exposure of Sensitive Information to an Unauthorized Actor
- CWE-201: Insertion of Sensitive Information Into Sent Data
- CWE-204: Observable Response Discrepancy
- CWE-208: Observable Timing Discrepancy
- CWE-209: Generation of Error Message Containing Sensitive Information
- CWE-214: Invocation of Process Using Visible Sensitive Information
- CWE-215: Insertion of Sensitive Information Into Debugging Code
- CWE-531: Inclusion of Sensitive Information in Test Code
- CWE-532: Insertion of Sensitive Information into Log File
- CWE-535: Exposure of Information Through Shell Error Message
- CWE-536: Servlet Runtime Error Message Containing Sensitive Information
- CWE-537: Java Runtime Error Message Containing Sensitive Information
- CWE-540: Inclusion of Sensitive Information in Source Code
- CWE-548: Exposure of Information Through Directory Listing
- CWE-550: Server-generated Error Message Containing Sensitive Information
- CWE-598: Use of GET Request Method With Sensitive Query Strings
- CWE-615: Inclusion of Sensitive Information in Source Code Comments

### How do I get set up? ###

#### Using Docker
1. 
```bash
docker build -t cwe-200 --build-arg JAVA_VERSION=<VERSION> . 
``` 
2. 
```bash
docker run -p 4200:4200 -p 5400:5400 cwe-200
```
3. http://localhost:4200/

#### Manual Install

#### Dependencies
1. [Node.js](https://nodejs.org/en?ref=altcademy.com) 20+ You will need to add Node as a system env variable
2. [CodeQL CLI](https://github.com/github/codeql-cli-binaries/releases) You will need to add CodeQL as a system env variable

Your project's build tool (Ensure that your project builds before using our tool)
[Maven](https://maven.apache.org/install.html), [Gradle](https://gradle.org/)

#### In the Root Directory

1. Install dependencies
```bash 
npm install
```

2. Build backend and frontend
```bash
npm run build
```

3. Setup Codeql dependencies 
```bash
npm run codeql-setup
```

4. Run the Front-end and Back-end Servers
```bash
npm run start
```
5. 
Navigate to http://localhost:4200/

### Query Testing ###
There is a labeled toy dataset located in `backend/Files/CWEToyDataset` that includes 333 test cases.
To test the results of the queries run this command `codeql database analyze backend/Files/CWEToyDataset/CWEToyDatasetdb --format=sarifv2.1.0 --output=out.sarif --rerun codeql/codeql-custom-queries-java` from the project root. 
After this is complete, navigate to `testing/toy-dataset-results.py` and run this file. You should see the accuracies printed out in the terminal. 
If you want to know exactly how each query classified their corresponding test cases. Look in the `codeql_results.json` file located in the same directory.

#### Making your queries work with the testing script ####
If you make a new query or modify one, make sure you add the @id CWE-<NUMBER> and @cwe CWE-<NUMBER> tags to the meta data at the top of the query.

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Repo owner or admin
* Other community or team contact

# Code Documentation 

## backend
The backend does most of the work in this project.

### codeql_setup
This is the code called when `npm run codeql-setup` is run in the root. Its purpose is to automate the setup of the CodeQL environment, including downloading the [workspace](https://github.com/github/vscode-codeql-starter) and the actual [CodeQL packages](https://github.com/github/codeql/tree/e27d8c16729588259f8143c7ed4569d517b0de10) and switching those packages to the correct version to work with the [CodeQL CLI Version 2.20.3](https://github.com/github/codeql-cli-binaries/releases/tag/v2.20.3). Along with copying over our queries from `/codeql queries` to `codeql/codeql-custom-queries-java`.

Three files are a part of this process:

<ul>
  <li>
    <code>run-platform-specific-setup.ts</code><br>
    Main wrapper that determines which platform is being used.
  </li>
  <li>
    <code>linux-setup.ts</code><br>
    Performs the setup specific to Linux.
  </li>
  <li>
    <code>windows-setup.ts</code><br>
    Performs the setup specific to Windows.
  </li>
</ul>

> **Note:** If you change the CLI version, you must update the git commit used in these files to have the matching lib files.

### Files
This is the directory where all of the projects are uploaded. Each project has its subdirectory within this, which holds the source files, CodeQL database, parsed source code (`parsedResults.json`), Attack Surface Detection Engine results (data.json), Exposure Analysis Engine results (`result.sarif`), and Flow Verification Engine results (`flowMapsByCWE.json`). You can use each of these to better understand why a detection appears in the output.

### ParseJava
This directory contains a small Maven project designed to parse Java source code and identify all instances of variables, string literals, comments, method calls, and method-level context. For example, if a variable `a` appears in the method `foo`, the parser creates a mapping from `a → foo`, enabling you to associate `a` with all code within the `foo` method. The extracted information is returned in a structured format, as shown below:

```json
{
  "filename": "file.java",
  "variables": [
    {
      "name": "a",
      "type": "int",
      "methods": ["foo"]
    }
  ],
  "comments": [
    {
      "name": "This is a comment",
      "methods": ""
    }
  ],
  "strings": [
    {
      "name": "This is a string",
      "methods": ["getName"]
    }
  ],
  "sinks": [
    {
      "name": "add",
      "methods": ["format"]
    }
  ],
  "methodCodeMap": {
    "getName": "Code for getName method",
    "format": "Code for format method"
  }
}
```
> **Note:** The first time a project is run for analysis, the JAR is automatically built, so there is no need to worry about building it manually.

### src
This directory is where the bulk of the tool’s code exists. There are nine subdirectories here that correspond to different parts of the tool, along with completely different (Some deprecated) approaches. If you want more information on how each component works, there are JavaDocs.

### analyze
Think of this component as the main wrapper for the entire tool. It is responsible for orchestrating all analysis stages across various modules: parsing, detection, injection, static analysis, and verification. When a project is sent to the backend via the /analyze/ endpoint, this component is executed.

If you're trying to understand the flow of the entire tool, this is the best place to set breakpoints and step through to see how each module interacts.

**AnalyzeService**
The AnalyzeService coordinates the full end-to-end pipeline for Java vulnerability analysis. The `runAnalysis()` method is the primary entry point. It performs the following steps:


1. **Java Setup**  
   a. Sets the appropriate Java version for the toolchain if specified.

2. **Attack Surface Detection Engine**  
   a. Collects all `.java` files in the uploaded project.  
   b. Parses them using ParseJava to extract variables, strings, comments, and method-level context.  
   c. Invokes the Sentence-BERT-based classifier to label potentially sensitive values.  
   d. Reads Sentence-BERT predictions from `data.json`.  
   e. Converts results into CodeQL-compatible `.yml` templates (`SensitiveVariables`, `SensitiveStrings`, `SensitiveComments`, `Sinks`) for injection.

3. **Exposure Analysis Engine**  
   a. Generates a CodeQL database for the uploaded source.  
   b. Runs customized CodeQL queries for selected CWEs.  
   c. Parses SARIF output and stores flow-level data in `flowMapsByCWE.json`.

4. **Flow Verification Engine**  
   a. If data flows are found, it runs Sentence-BERT-based verification to determine if each flow is valid.  
   b. Updates the SARIF file with verification labels.

5. **Output Results**  
   a. Returns results either in SARIF or CSV format, based on the requested extension.


### bert
This component implements the main component of the Attack Surface Detection Engine. While it has a bert.service.ts file. This file is mostly used to call and parse the results of the Python files. Here is what each of the main Python files is responsible for:


#### inference
##### `attack_surface_detection.py`
This script powers the Sentence-BERT-based Attack Surface Detection Engine, performing contextual analysis on variables, strings, comments, and sinks extracted from Java source code. It reads the method-level parsed output (`parsedResults.json`) from ParseJava, applies deep learning models to classify potentially sensitive elements, and outputs the results as data.json.

**Main Responsibilities**
- Loads a pre-trained Sentence-BERT model for embedding code identifiers and context.
- Applies torch.jit-compiled PyTorch models (one per data type: variables, strings, comments, sinks).
- Produces predictions indicating whether elements are security-sensitive (or, for sinks, what sink type they are).
- Outputs structured predictions in `data.json`, aligned by file name and element type.

**Pipeline Overview**
1. **Data Loading**  
   a. Asynchronously reads `parsedResults.json`, which includes code elements and their surrounding method contexts.

2. **Preprocessing**  
   a. Applies CamelCase splitting and lowercasing (`text_preprocess`).  
   b. Converts each element into two fields:  
      i. `preprocessed_item` (identifier name)  
      ii. `context` (surrounding method text)

3. **Encoding**  
   a. Uses Sentence-BERT to embed both name and context fields. For variables, strings, and sinks, the vectors are concatenated into a single vector per element.

4. **Inference**  
   a. Each data type is run through a dedicated PyTorch model:  
      i. Binary classification for variables, strings, and comments  
      ii. Multi-class classification for sinks (e.g., I/O Sink, Log Sink, Network Sink)

5. **Threshold Filtering**  
   a. Only predictions exceeding the defined confidence thresholds are included in the final output.

6. **Result Output**  
   a. Writes a JSON list to `data.json`, structured like:

```json
[
  {
    "fileName": "Example.java",
    "variables": [
      {
        "name": "password",
        "confidence": "0.87"
      }
    ],
    "sinks": [
      {
        "name": "System.out.println",
        "type": "Print Sink",
        "confidence": "0.91"
      }
    ]
  },
  ...
]
```

> **Note:**
> - Parallel Batch Embedding:
Embedding is done in parallel using concurrent.futures.ThreadPoolExecutor.
> - Device-Aware Execution:
Automatically uses GPU (CUDA) if available; otherwise falls back to CPU.
> - Custom Thresholds:
Confidence cutoffs are defined per type (e.g., strings: 0.95, variables: 0.75).
Feel free to adjust these if you train the models with more data. Just make sure reducing these doesn’t increase false positives.

**How It Fits In?**
This script is invoked by the analyze service, which:
- Runs ParseJava to get `parsedResults.json`
- Then executes `attack_surface_detection.py` as a subprocess
- Uses the output `data.json` to inject findings into CodeQL queries

##### `flow_verification.py`
This script performs flow-level verification on the results from the Exposure Analysis Engine. using embeddings from GraphCodeBERT. It takes as input a flowMapsByCWE.json file containing data flows for CWE violations, and outputs an updated JSON file where each flow is labeled as "Yes" or "No" based on model inference.

> **Note:** This approach is similar to `attack_surface_detection.py`, however, for dataflows instead of attack surfaces. One major difference is the use of an aggerator, because GraphCodeBERT only has a context window of 512 tokens. This aggregator takes overlapping chunks and aggregates them into a single 512 token vector to avoid truncation. This is important for long flows, where what we are looking for is to see if the sensitive information is exposed in the last flow node. 

**Responsibilities:**
- Load and parse SARIF-derived data flows (`flowMapsByCWE.json`)
- Format data flow traces into strings compatible with transformer encoders
- Generate contextual embeddings using GraphCodeBERT with a custom aggregator.
- Run binary classification using a pre-trained TorchScript model (`verify_flows_final_model.pt`)
- Append predicted labels (label) and confidence scores (probability) to the original data flows.
- Update results in `flowMapsByCWE.json` to now have a label associated with each flow.

**Architecture**

1. **Embedding**  
    a. Utilizes `microsoft/graphcodebert-base` to embed textual representations of data flows  
    b. Handles long sequences by segmenting >512 tokens and aggregating via:  
        i. TransformerAggregator: Transformer Encoder with mean-pooling  

2. **Inference**  
    a. Loads a TorchScript-compiled model (`verify_flows_final_model.pt`) for classification  
    b. Returns binary predictions (Yes/No) based on a 0.5 probability threshold  

3. **Postprocessing**  
    a. Each flow entry is updated with:  

    ```json
    {
      "label": "Yes",
      "probability": 0.92
    }
    ```

**Inputs**
- `flowMapsByCWE.json`: JSON file mapping CWE → flow → code steps
- `verify_flows_final_model.pt`: TorchScript model trained to classify valid data flows
- `aggregator.pt`: Pre-trained aggregator weights used to merge long GraphCodeBERT embeddings

**Output**
`flowMapsByCWE.json` → Overwritten or copied with new fields: label, probability


#### training
This directory contains the full training pipeline for building classification models that power the Attack Surface Detection Engine and Flow Verification Engine. These models determine whether variables, strings, comments, or sinks are sensitive or security-relevant, based on their name and context.

##### `train_attack_surface_models.py`
This script trains a series of binary or multi-class PyTorch classifiers using various embedding models (e.g., Sentence-BERT, CodeT5, CodeBERT). The classifiers are trained using labeled datasets that combine name and context information extracted from Java code.

**Responsibilities:**
- Load labeled training data (`backend/src/bert/training/data/labels.json`) and context (`backend/src/bert/training/data/parsedResults.json`).
    > - **Note:** 
    >   - If you add more labeled training data, you must run that data through the ParseJava component to get the correct structure. 
    >   - You don’t need to label every file in the training data. If you have, for example, 2000 files, but you only labeled one file, only that one file will be used. A check is performed to make sure it only uses data that has been labeled. 
- Embed identifier names and method-level context using Sentence-BERT, CodeT5, CodeBERT, or Longformer.
- Perform random search hyperparameter tuning over a parameter grid.
- Train separate models for each data type:
    - variables (binary)
    - strings (binary)
    - comments (binary)
    - sinks (multi-class)
- Evaluate each model and log metrics, including accuracy, precision, recall, F1, and the confusion matrix.
- Save the final models in TorchScript format (`*.pt`) to disk.

**Architecture:**
Each model consists of a deep feed-forward neural network with:
- Multiple linear layers (4, with skip layers)
- Batch normalization
- Dropout
- Residual connections (for a binary classifier)
- Separate model classes are defined for:
    - BinaryClassifier: used for variables, strings, and comments
    - MultiClassClassifier: used for sinks

Each embedding combines name and context vectors using concatenation.

!!! Important:** The new models will be saved to the `/models` directory in the root. You must paste the new models in `backend/src/bert/models` and update the name to use it. This is done so you don’t overwrite the current models if the training worsens. For example if you have `model/sentbert_best_model_variables.pt`. You must rename it to `variables.pt` and paste it in `backend/src/bert/models`.


**Output**
- Trained models saved as TorchScript files (e.g., `sentbert_final_model_variables.pt`)
- Evaluation results written to:
    - `backend/src/bert/training/results.txt`
- This includes metrics for every model/category combination.

**Integration Notes**
- The models trained here are later used by:
    - `attack_surface_detection.py` (Attack Surface Detection Engine)

##### `train_flow_verification_model.py`
This script trains a GraphCodeBERT-based classifier to verify SARIF data flows by classifying them as either sensitive (**"Yes"**) or non-sensitive (**"No"**). It performs preprocessing, embedding, model definition, hyperparameter tuning, evaluation, and TorchScript model export.

**Responsibilities:**
- Load and deduplicate labeled data flows from a directory.
- Preprocess flow sequences into natural language format for embedding.
- Use GraphCodeBERT to generate contextual vector embeddings.
- Aggregate long sequences using a custom TransformerAggregator.
- Define a multi-layer residual PyTorch classifier (ClassifierModel).
- Run randomized hyperparameter search using skorch + - TqdmRandomizedSearchCV.
- Save the best-performing model as a TorchScript `.pt` file.

**Workflow Summary:**
1. **Load & Preprocess**
    a. Reads all `.json` files from `testing/Labeling/FlowData`.
    b. Deduplicates flows using SHA256 hash of their string representation.
    c. Each flow is labeled as 1 (Yes) or 0 (No).
2. **Embedding**
    a. Flow strings are embedded using `Microsoft/graphcodebert-base`.
    b. Long inputs (>512 tokens) are segmented and aggregated via TransformerAggregator.
3. **Model Architecture**
    a. ClassifierModel: A dense feedforward neural net with batch norm, dropout, and residual-style layers.
    b. Output is a sigmoid-activated binary prediction.
4. **Training**
    a. Uses skorch.NeuralNetClassifier for PyTorch + sklearn compatibility.
    b. Hyperparameters are tuned with RandomizedSearchCV over:
        i. Learning rate, dropout, activation, batch size, epochs.
    c. SMOTE is used to rebalance the training set.
5. **Evaluation**
    a. Prints detailed precision, recall, F1 score, confusion matrix.
    b. Saves the final model as `verify_flows_final_model.pt` in TorchScript format.

**Input:**
- Directory: `testing/Labeling/FlowData/`
- Contains `.json` files with flows like:
```json
{
  "CWE-200": [
    {
      "resultIndex": 0,
      "fileName": "Example.java",
      "flows": [
        {
          "codeFlowIndex": 0,
          "flow": [...],
          "label": "Yes"
        }
      ]
    }
  ]
}
```

**Output:**
- `verify_flows_final_model.pt`: TorchScript classifier saved to `backend/src/bert/models/`
- `aggregator.pt`: Transformer aggregator weights (used during inference)
- Printed metrics: Precision, Recall, F1, Accuracy, AUC, and classification report

**Use Case:**
This script is a flow verifier for dataflow-based SARIF outputs. It can be used to filter false positives from CodeQL results by training a classifier that learns patterns of valid sensitive flows across our covered CWEs.

### chat-gpt
This code is responsible for the original approach that used Chat-GPT for the Attack Surface Detection Engine. This code is deprecated as BERT is now used. However, it is kept in case the approach is ever needed again, either for comparison or for being updated to work better. In the future, feel free to delete this if it is determined to never be needed again. 

### code-ql
This directory handles the integration and orchestration of CodeQL analysis within the system. It is responsible for running CodeQL commands, parsing the SARIF output, filtering false positives, and transforming data into a format usable by other components like the Flow Verification Engine and PIQUE.

It consists of two main files:

##### `code-ql.service.ts`
This service acts as the controller for all CodeQL-related operations. It executes CodeQL CLI commands, creates databases, runs custom CWE queries, and manages the post-processing of SARIF outputs. 

**Code Overview:**

- **`createDatabase():`** 
    - Initializes a CodeQL database for a given Java project.
- **`runCWEQueries():`**
    - Executes a suite of custom CWE queries (custom-codeql-queries) on the database and writes results to a SARIF file.
- **`getSarifResults():`** 
    - Retrieves and parses SARIF results using the parser service.
- **`getDataFlowTree():`** 
    - Extracts flow trees for specific vulnerabilities and is used for visual inspection and verification.
- **`labelFlows():`** 
    - Integrates manual flow labeling (e.g., from a front-end UI) into the `flowMapsByCWE.json` file.
- **`updateSarif():`** 
    - Applies the verified flow labels back into the SARIF file, enabling downstream tools to consume cleaned, labeled results.
- **`performBackwardSlicing() [deprecated]:`** 
    - Supports backward program slicing, which is currently unused in the main workflow.

This file interacts directly with the CodeQL CLI using child_process.spawn and streams real-time output to the frontend via a WebSocket gateway (EventsGateway).


##### `codql-parser-service.ts`
This service parses, filters, and restructures the SARIF output from CodeQL to make it usable by other components in the system.

**Code Overview:**
- **`getSarifResults():`** 
    - Loads and filters the SARIF results (e.g., removes flows not labeled as "yes").
- **`getDataFlowTree():`** 
    - Constructs a detailed, human-readable representation of each flow by loading relevant code snippets from source files.
- **`saveDataFlowTree():`** 
    - Writes all flow data grouped by CWE into `flowMapsByCWE.json` for use by the Flow Verification Engine.
- **`parseRules() / parseResults():`** 
    - Organize SARIF data by rule and file, which is used by the frontend and for indexing.
- **`getcsvResults():`**
    - Converts filtered SARIF results into a simplified CSV format for consumption by the PIQUE evaluator.

This file is central to interpreting the SARIF format and integrating human-readable and model-verifiable formats across the system. All code flow locations are resolved to the surrounding source code for added context.

### events
This directory contains the WebSocket gateway used for real-time communication between the backend and connected clients (typically the frontend).

##### `events.gateway.ts`
The EventsGateway is a WebSocket gateway that enables the backend to emit live updates during long-running or asynchronous operations, such as CodeQL analysis. This ensures that users receive progress updates without needing to manually refresh or poll the server. 

**Code Overview:**
- **`handleConnection():`** 
    - Logs when a client connects to the WebSocket server.
- **`handleDisconnect():`** 
    - Logs when a client disconnects from the server.
- **`handleMessage():`** 
    - Responds to the 'message' event sent by a client. Currently emits a "Hello everyone!" message to all clients.
- **`emitDataToClients(type, data):`** 
    - Utility function used throughout the backend (e.g., during CodeQL execution) to broadcast analysis progress, errors, or status messages to all connected clients over the 'data' channel.

This gateway is used to stream terminal output from child processes (like CodeQL CLI commands) directly to the user interface.


### files
This component handles all file-level operations, including uploading, extracting, parsing, reading, writing, and organizing Java project files. It is tightly integrated with other components, enabling them to access project data, manipulate files, and dynamically manage Java versions.

##### `files.service.ts`
The FilesService acts as the interface for handling file uploads and retrievals. It wraps core file operations and interacts with the real-time WebSocket gateway (EventsGateway) to notify the frontend about progress or results. 

**Key responsibilities:**
- **`create(file):`** 
    - Takes an uploaded .zip file containing Java source code, saves and extracts it to the configured project directory, and returns the directory structure as a tree.
- **`findOne(fileId):`** 
    - Reads and returns the content of a specific file based on its path, allowing on-demand code preview in the frontend or during analysis.

##### `fileUtilService.ts`
The FileUtilService is a comprehensive utility library for file system interactions across the backend. It supports everything from zip extraction and file parsing to Java environment configuration. It is marked as a global provider to be used across the application. 

**Code Overview:**
- **`writeZipFile(file):`** 
    - Extracts the contents of a `.zip` upload and places it in the correct project directory.
- **`getDirectoryTree(dirPath):`** 
    - Recursively builds and returns a nested JSON-like tree structure representing the folder and file hierarchy.
- **`getJavaFilesInDirectory(dirPath):`** 
    - Finds all .java files in a directory and its subdirectories.
- **`readFileAsync(path) and readJsonFile(path):`** 
    - Read and return file contents as a string or parsed JSON, respectively.
- **`writeToFile(path, content):`** 
    - Writes content to the specified file path using UTF-8 encoding.
- **`parseJSONFile(path):`** 
    - Parses the data.json output from the Attack Surface Detection Engine and returns structured lists of sensitive variables, strings, comments, and sinks, along with per-file mappings.
- **`processJavaFile(path, id): [Unused]`** 
    - Cleans and processes a Java file by removing blank lines and adjusting the structure for easier analysis or model input. Previously Used with our deprecated ChatGPT approach. 

- **`removeDir(path):`** Deletes a directory and its contents recursively.
- **`processFilePath(sourcePath, filePath) and getFilenameFromPath(filePath):`** 
    - Utility functions to ensure consistent and platform-independent path handling.
- **`convertLabeledDataToMap(labeledData):`** 
    - Converts labeled data from JSON into a nested map structure for easier querying by file and type.
- **`saveToJsonl(filePath, data):`** 
    - Saves structured data to a .jsonl (JSON Lines) file format for use in training or analysis workflows.
- **`setJavaVersion(version, useAnyJava):`** 
    - Automatically configures the correct Java version for the environment. On Windows, uses winget to install Azul Zulu builds. On Linux/macOS, downloads and installs from Adoptium based on the version mapping.


### java-parser
This component is responsible for integrating the ParseJava JAR, which performs static extraction of key program elements from Java source files. It wraps the JAR execution, orchestrates parsing across all files in a project, and aggregates the output into a single JSON file. 

**Code Overview:** 
- **`wrapper(filePaths, sourcePath):`** 
    - The main entry point that:
        1. Builds the parser JAR if necessary.
        2. Parses all .java files in the project concurrently.
        3. Emits parsing progress in real-time via WebSockets.
        4. Saves the parsed results to `parsedResults.json`.

- **`getParsedResults(filePaths):`** 
    - Uses a worker pool to concurrently parse Java files. Updates parsing progress as each file completes.
- **`buildJarIfNeeded():`** 
    - Builds the Maven project under `ParseJava/` using `mvn clean package` if the JAR file does not already exist.
- **`parseJavaFile(filePath):`** 
    - A wrapper around the Java program that invokes the parser JAR on a single file.
- **`runJavaProgram(jarPath, filePath):`** 
    - Runs the JAR using java -jar and parses the output (JSON). Returns structured JavaParseResult objects. If parsing fails, it returns an empty result with only the filename.

The output of this component (stored in `parsedResults.json`) is critical to the Attack Surface Detection Engine and later stages of the vulnerability analysis pipeline.


### llm
This component is similar to the chat-gpt component. It was meant to serve as a more general-purpose LLM approach. Currently, it is set up to use Llama, which was hosted on a JetStream server. This is mainly in a proof-of-concept phase, so the endpoint is just hardcoded and is no longer up, as it was deemed not to work that well. Just like the chat-gpt component, this is not used and is kept in the code in case it is needed in the future. 

### templates
This directory holds predefined YAML templates used to inject the Attack Surface Detection Engine’s results into the CodeQL query pipeline. These templates are required to format and populate data in a way that CodeQL's extensible queries can consume.

These files are used during the `saveSensitiveInfo()` process inside the analyze component. They dynamically generate `.yml` files for CodeQL by embedding sensitive variables, strings, comments, and sinks detected in the source code. 

**Templates Overview:**

##### **`SensitiveVariables.ts`**
Defines the YAML structure for injecting detected sensitive variables into CodeQL via:
```json
export const SensitiveVariables:string=`
extensions:
  - addsTo:
      pack: custom-codeql-queries
      extensible: sensitiveVariables
    data:
    - ["fileName", "sensitiveVariable"]
---------- 
```

The `----------` marker is replaced with the variable names found in the Attack Surface Detection Engine.

##### **`SensitiveStrings.ts`**
Defines the structure for sensitive string literals:
```json
extensions:
  - addsTo:
      pack: custom-codeql-queries
      extensible: sensitiveStrings
    data:
    - ["fileName", "senstiveString"]
++++++++++
```
The `++++++++++` marker is replaced with the string values found in the Attack Surface Detection Engine


##### **`SensitiveComments.ts`**
Specifies the format for injecting suspicious comments:
```json
extensions:  
  - addsTo:
      pack: custom-codeql-queries
      extensible: sensitiveComments
    data:
    - ["fileName", "senstiveComment"]
**********
```
The `**********` marker is replaced with the comments found in the Attack Surface Detection Engine

##### **`Sinks.ts`**
Used to inject sink data (e.g., method names and types that represent sensitive information flows):

```json
extensions:
  - addsTo:
      pack: custom-codeql-queries
      extensible: sinks
    data:
    - ["fileName", "sinkName", "type"]
----------
```
The `----------` marker is replaced with the method names  and type found in the Attack Surface Detection Engine.


##### **`data.ts`**
An older and currently unused format (used by the Chat-GPT component) for injecting data directly into `.ql` query files using inlined JavaScript template strings. It includes template code for extending CodeQL classes such as SensitiveVariable, SensitiveStringLiteral, and SensitiveComment. This is kept for reference and legacy support.

These templates ensure the results from the Attack Surface Engine are seamlessly translated into CodeQL-compatible formats, enabling end-to-end hybrid analysis between machine learning and static analysis.

## codeql
This directory is automatically created when codeql_setup is run via npm run codeql-setup. Our queries are copied from the code queries directory into `/codeql-custom-queries-java`. That is really all you need to worry about in this directory, as the directories for other programming languages are unused. 

Note: If you update the queries in code queries you must delete the queries in `/codeql-custom-queries-java` and `run npm run codeql-setup` again to copy over the changes. Make sure you make all your query updates in the code queries directory, not this one, as this isn’t tracked by Git.

## codeql queries 
This directory holds all of the queries that the Exposure Analysis Engine uses. They are organized into their own sub-directories, and it is clear which query belongs to which CWE. 

There is also some supporting code in this directory:
- **`/Barrier`** 
    - **`Barrier.qll`**
        - It contains a big list of common barriers that tell the queries that if the data flows into one of the defined barriers, the flow will no longer be considered. For example, if data flows into a masking function.
        - > **Note:** All of the queries use this.
- **`/CommonSinks`**
    - **`CommonSinks.qll`**
        - Contains a list of common sinks sorted by these types
            - Logging
            - Servlet 
            - Print
            - Error
            - IO
            - Spring
        - The purpose of this is to just add some baseline sinks that can be used by specific queries, depending on their coverage. These are no meant to be exhaustive and the Attack Surface Detection Engine is responsible for the majority of the sinks used.
- **`/SensitiveInfo`**
    - This is where all of the Attack Surfaces `.yml` files that were written in `backend/templates` are stored. 
- **`/SensitiveInfo.qll`**
    - This defines a set of predicates that are used to bridge the gap between the yml files and the actual queries. This is a very important file because all of the queries use the predicates defined in this file. 

## frontend
WIP

## testing
The `/testing` directory contains all scripts and data used to evaluate, label, and benchmark components of the Attack Surface Detection Engine and Flow Verification Engine, as well as various experimental and legacy efforts. It is organized by functional purpose

### Advisory
This script aims to get all the CWE-200 security advisories related to Java on GitHub. It checks for projects with Maven since Java isn’t an option. The output is saved in `top_advisories.txt`. About 85% of the projects are correct. Some are either not Java projects at all or use Java very little. However, this was the best method used to get Java projects for testing. If you are looking for more Java projects, just increase the number from 150. 

> **Note:** This requires a GitHub token that you can generate with your GitHub account. Just make sure not to commit any changes to the script while your token is still in the code. Feel free to update it to be an arg instead of hardcoding it if needed. It won’t break anything.

### Call Graph [Unused]
This script is an experimental approach that is no longer used. It was meant to work with the CodeQL query in `WIP queries/ProgramSlicing/Sinks/CallGraph.ql` by parsing the output of that query into a more human readable format that would then be passed to an ML model sort of how the Flow Verification Engine works, but on the Attack Surface Detection Engine outputs. This code is kept in case that approach is ever needed again. It is still not fully complete.

### ChatGPT [Unused]
This directory contains data and scripts related to measuring ChatGPT’s performance on attack surface detection. You can think of the work in this directory as belonging to two main functions: Data Labeling / Preprocessing and ChatGPT Approach Measuring. All of this data is no longer used, but left in case it is ever needed again. Feel free to delete it.
 
To understand this best, it’s essential to know that most of the original labeling was done on the CVE dataset in the `backend/Files/CVEDataset` in a Google Sheet between Sara and David. This was initially done for just variables, strings, and comments. Later on, however, sinks were added, but they were not done in the same format. So the parsing and combining is done across a few scripts, and it is messier than what it should be. The data pertaining to this is, 

- **`Peer Review Data Files.xlsx`** 
    - The labeling for each CVE file for all variables, strings, and comments.
- **`parse_csv.py`**
    - Used to parse these labels into a JSON that matches ChatGPT’s output format.
- **`Sinks.xlsx`**
    - The labeled sinks for the CVE files.
- **`sink_parser.py`**
    - Parses the labeled sinks into `labeled_sinks.json`.

The rest of the data here relates to measuring the actual approach for the CVE dataset and the ToyDataset.

- **`classification.py:`**
    - This script is used to compare the classifications made by ChatGPT to the actual CWEs of the files in the CWEToyDataset. We mainly want to know if just using ChatGPT for classification is viable and if it can beat our current results.
    - This is was used to see if ChatGPT could be used for everything. Basically, given a file, what are the vulnerabilities within it? Used for experiments.

- **`toy_dataset_labeled_score.py:`**
    - Used to see how ChatGPT performed at detecting attack surfaces in our CWEToyDataset.

- **`peer_review_parser.py:`**
    - A more advanced parser that handles multiple reviewers and sheets. While also calculating ChatGPT’s performance at detecting attack surfaces in our CVE dataset.

- **`combine_formats.py:`** 
    - Used to turn our labels into training data for ChatGPT. 
    - This was another approach that we tried where we wanted to see if fine-tuning GPT-4o helped improve the performance.



