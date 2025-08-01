/**
* MIT License
* Copyright (c) 2019 Montana State University Software Engineering Labs
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/
package tool;

import org.json.JSONException;
import org.json.JSONObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import pique.analysis.ITool;
import pique.analysis.Tool;
import pique.model.Diagnostic;
import pique.model.Finding;
import utility.HelperFunctions;
import pique.utility.PiqueProperties;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.ThreadLocalRandom;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import org.json.JSONArray;
import org.json.JSONObject;

/**
 * CODE TAKEN FROM PIQUE-BIN-DOCKER AND MODIFIED FOR PIQUE-SBOM-CONTENT and
 * PIQUE-CLOUD-DOCKERFILE.
 * This tool wrapper will run and analyze the output of the tool.
 * When parsing the output of the tool, a command line call to run a Python
 * script is made. This script is responsible for translating from
 * CVE number to the CWE it is categorized as by the NVD.
 * 
 * @author Derek Reimanis
 *
 */
public class CweCodeQl extends Tool implements ITool {
    private static final Logger LOGGER = LoggerFactory.getLogger(CweCodeQl.class);
    private String backendAddress;
    private String projectName;
    private Path outputFilePath;
    private boolean serverOnline = false;

    public CweCodeQl(String backendAddress) {
        super("CweCodeQl", null);
        this.backendAddress = backendAddress;
        this.serverOnline = this.isServerRunning(this.backendAddress);

        // Read in all of the project's information
    }

    // Methods
    /**
     * @param projectLocation The path to a binary file for the desired solution of
     *                        project to analyze
     * @return The path to the analysis results file
     */
    @Override
    public Path analyze(Path projectLocation) {
        this.projectName = projectLocation.getFileName().toString();
        if (this.projectName.endsWith(".zip")) {
            this.projectName = this.projectName.substring(0, this.projectName.length() - 4); // remove ".zip"
        }

        if (this.projectName == "projects") {
            LOGGER.info(projectName + " is a directory, not a project. Make sure you are running from the wrapper.");
            return null;

        }
        System.out.println("\n");
        LOGGER.info("Analyzing " + this.projectName);

        // set up results dir

        String workingDirectoryPrefix = "";

        try {
            // Load properties
            Properties prop = PiqueProperties.getProperties("src/main/resources/pique-properties.properties");
            Path resultsDir = Paths.get(prop.getProperty("results.directory"));

            // Set up working directory
            workingDirectoryPrefix = resultsDir + "/tool-out/CWE-200/";
            Files.createDirectories(Paths.get(workingDirectoryPrefix));
            // Set up output file path
            this.outputFilePath = Paths.get(workingDirectoryPrefix, this.projectName + ".csv");

            // System.out.println("Output file path: " + this.outputFilePath);

        } catch (IOException e) {
            e.printStackTrace();
            LOGGER.debug("Error creating directory to save CweCodeQl tool results");
            System.out.println("Error creating directory to save CweCodeQl tool results");
        }

        return runCWE200Tool(workingDirectoryPrefix, projectLocation);

        // return null;

    }

    /**
     * parses output of tool from analyze().
     *
     * @param toolResults location of the results, output by analyze()
     * @return A Map<String,Diagnostic> with findings from the tool attached.
     *         Returns null if tool failed to run.
     */
    @Override
    public Map<String, Diagnostic> parseAnalysis(Path toolResults) {
        // Just for testing, remove hardcoded path later
        toolResults = Paths.get(this.outputFilePath.toUri());

        LOGGER.info(this.getName() + " Parsing Analysis...");

        Map<String, Diagnostic> diagnostics = HelperFunctions.initializeDiagnostics(this.getName());
        List<String[]> results = null;

        try {
            results = HelperFunctions.readFileContent(toolResults);
            LOGGER.info("Number of detections found: " + results.size());
            for (String[] record : results) {
                String cweId = record[0];
                String filePath = record[1];
                int lineNumber = Integer.parseInt(record[2]);
                int characterNumber = Integer.parseInt(record[3]);
                int severity = this.cweToSeverity(cweId);

                Diagnostic diag = diagnostics.get("CWE-" + cweId + " Diagnostic CweCodeQl");
                if (diag != null) {
                    Finding finding = new Finding(filePath,
                            lineNumber,
                            characterNumber,
                            severity);
                    finding.setName(cweId);
                    diag.setChild(finding);
                    


                    // System.out.println(cweId + " " + cweDescription + " " + filePath + " " +
                    // lineNumber + " " + characterNumber + " " + severity);
                }

            }

            // LOGGER.info("Diagnostics parsed: " + diagnostics.size());
            diagnostics.forEach((k, v) -> LOGGER.info(k + ": " + v.getValue()));

        } catch (IOException e) {
            LOGGER.info("No results to read from CweCodeQl.");
        }
        return diagnostics;
    }

    private Path runCWE200Tool(String workingDirectoryPrefix, Path projectLocation) {
        // Check if the results file already exists
        if (!doesExist(workingDirectoryPrefix)) {

            // Check to see if the server is running
            if (this.serverOnline) {

                // Upload the project to the server
                try {
                    uploadProjectToServer(backendAddress, projectLocation);
                } catch (Exception e) {
                    LOGGER.error("Failed to upload project to server at: " + backendAddress);
                    e.printStackTrace();
                    return null;
                }

                String javaVersion = "11";
                try {
                    // Get the Java version for the specific project
                    Path projectInfoFilePath = Paths.get("..", "testing", "PIQUE_Projects", "projects.json");
                    javaVersion = HelperFunctions.getJavaVersion(projectInfoFilePath, this.projectName);
                } catch (Exception e) {
                    LOGGER.error("Failed to get Java version from server at: " + backendAddress);
                    e.printStackTrace();
                    return null;
                }

                // Perform the analysis
                JSONObject jsonResponse = null;
                try {
                    LOGGER.info("CWE-200 tool is analyzing this might take a while.");
                    String toolResults = sendPostRequestToServer(backendAddress, this.projectName, javaVersion);
                    jsonResponse = HelperFunctions.responseToJSON(toolResults);

                } catch (Exception e) {
                    LOGGER.error("Failed to send POST request to server at: " + backendAddress);
                    e.printStackTrace();
                    return null;
                }

                try {
                    if (jsonResponse.has("error") && !jsonResponse.isNull("error")) {
                        LOGGER.error("Error running analysis");
                        return null;
                    } else {
                        LOGGER.info("Analysis completed successfully");
                    }
                } catch (Exception e) {
                    return null;
                }

                // Convert the results to a CSV file, and save it
                Path finalResults = saveToolResultsToFile(jsonResponse, this.outputFilePath);
                return finalResults;

            } else {
                // TODO: Start the server
                LOGGER.error("Server is not running at: " + backendAddress);
                return null;
            }
        }
        // Something went wrong

        return Paths.get("output/tool-out/CWE-200/"+this.projectName+".csv");

    }

    private int cweToSeverity(String cweId) {
        switch (cweId) {
            case "CWE-201":
                return getRandomSeverity(8, 10); // Critical range
            case "CWE-208":
                return getRandomSeverity(6, 8); // High range
            case "CWE-209":
                return getRandomSeverity(5, 7); // Medium range
            case "CWE-214":
                return getRandomSeverity(8, 10); // Critical range
            case "CWE-215":
                return getRandomSeverity(6, 8); // High range
            case "CWE-531":
                return getRandomSeverity(5, 7); // Medium range
            case "CWE-532":
                return getRandomSeverity(8, 10); // Critical range
            case "CWE-535":
                return getRandomSeverity(5, 7); // Medium range
            case "CWE-536":
                return getRandomSeverity(8, 10); // Critical range
            case "CWE-537":
                return getRandomSeverity(5, 7); // Medium range
            case "CWE-538":
                return getRandomSeverity(8, 10); // Critical range
            case "CWE-540":
                return getRandomSeverity(8, 10); // Critical range
            case "CWE-548":
                return getRandomSeverity(6, 8); // High range
            case "CWE-550":
                return getRandomSeverity(8, 10); // Critical range
            case "CWE-598":
                return getRandomSeverity(8, 10); // Critical range
            case "CWE-615":
                return getRandomSeverity(6, 8); // High range
            default:
                return getRandomSeverity(3, 5); // Default low range
        }
    }

    private int getRandomSeverity(int min, int max) {
        return ThreadLocalRandom.current().nextInt(min, max + 1);
    }

    private boolean isServerRunning(String backendAddress) {
        try {
            String[] cmd = { "curl", backendAddress };
            HelperFunctions.getOutputFromProgram(cmd);
        } catch (Exception e) {
            LOGGER.error("Server is not running at: " + backendAddress);
            return false;
        }
        LOGGER.info("Server is running at: " + backendAddress);
        return true;
    }

    private void uploadProjectToServer(String backendAddress, Path zipPath) {
        try {
            // Define the command array with properly escaped JSON
            String[] cmd = {
                    "curl",
                    "-X", "POST",
                    "-F", "file=@" + zipPath.toString(),
                    backendAddress + "/files"
            };

            // Execute the command
            HelperFunctions.getOutputFromProgram(cmd);
        } catch (Exception e) {
            LOGGER.error("Failed to upload project to server at: " + backendAddress);
            e.printStackTrace();
        }
    }

    private String sendPostRequestToServer(String backendAddress, String projectName, String javaVersion) {
        String toolResults = "";
        try {
            // Build the JSON payload using a JSON library
            JSONObject json = new JSONObject();
            json.put("project", projectName);
            json.put("extension", "csv");
            json.put("format", "csv");
            json.put("javaVersion", javaVersion);
            String jsonData = json.toString(); // e.g.
                                               // {"project":"myProject","extension":"csv","format":"csv","javaVersion":"17"}

            // Append the endpoint to the backend address
            URL url = new URL(backendAddress + "/analyze/");
            HttpURLConnection connection = (HttpURLConnection) url.openConnection();
            connection.setRequestMethod("POST");
            connection.setRequestProperty("Content-Type", "application/json; charset=UTF-8");
            connection.setDoOutput(true);

            // Write the JSON payload to the output stream
            try (OutputStream os = connection.getOutputStream()) {
                byte[] input = jsonData.getBytes(StandardCharsets.UTF_8);
                os.write(input, 0, input.length);
            }

            // Get the response code and choose the appropriate input stream
            int responseCode = connection.getResponseCode();
            InputStream inputStream;
            if (responseCode < HttpURLConnection.HTTP_BAD_REQUEST) {
                inputStream = connection.getInputStream();
            } else {
                inputStream = connection.getErrorStream();
            }

            // Read the response line by line
            StringBuilder responseBuilder = new StringBuilder();
            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(inputStream, StandardCharsets.UTF_8))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    responseBuilder.append(line.trim());
                }
            }
            toolResults = responseBuilder.toString();

            connection.disconnect();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return toolResults;
    }

    private Path saveToolResultsToFile(JSONObject jsonResponse, Path outputFilePath) {
        try {
            // Parse JSON response to extract "data" field
            String data = jsonResponse.getString("data");

            // Clean up the CSV string if it has extra quotes at the beginning or end
            // data = data.replaceAll("^\"|\"$", ""); // Remove surrounding quotes if
            // present
            Path outputPath = Paths.get(outputFilePath.toUri());
            // Write the extracted data to the specified file path
            Files.write(outputPath, data.getBytes());

            LOGGER.info("Tool results saved to file");
            return outputPath;
        } catch (IOException e) {
            LOGGER.error("Failed to save tool results to file: " + outputFilePath);
            e.printStackTrace();
        } catch (Exception e) {
            LOGGER.error("Failed to parse JSON or process tool results.");
            e.printStackTrace();
        }
        return null;
    }

    private boolean doesExist(String workingDirectoryPrefix) {
        File tempResults = new File(workingDirectoryPrefix  + this.projectName + ".csv");
        if (tempResults.exists()) {
            LOGGER.info(
                    "Already ran CweCodeQl on: " + this.projectName + ", results stored in: " + tempResults.toString());
            return true;
        }
        tempResults.getParentFile().mkdirs();
        return false;
    }

    /***
     * Note from Derek ---- Usually we use this method to test to see if the tool
     * has been initialized properly. The code
     * I removed was voilerplate code for running a program with a '-version'
     * argument to make the the tool is set up properly.
     * Because codeql runs on a local server, you might want to override this method
     * so it pings the server, and initializes
     * it if it is not initialized. I removed the main() function from this class
     * too, the code in the main function appeared
     * to be the initialization code
     * 
     * @param toolRoot
     *                 The initial location of this tool's root folder.
     * @return
     */
    @Override
    public Path initialize(Path toolRoot) {
        return null;
    }
}
