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

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import pique.analysis.ITool;
import pique.analysis.Tool;
import pique.model.Diagnostic;
import pique.model.Finding;
import pique.utility.PiqueProperties;
import utilities.helperFunctions;

import java.io.File;
import java.io.IOException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.StandardCopyOption;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

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

    public static void main(String[] args) {
        CweCodeQl test = new CweCodeQl(PiqueProperties.getProperties().getProperty("backend.server"));
        // test.initialize(null);
        test.analyze(Paths.get("SmallTest"));
        // test.analyze(Paths.get("s"));


    }


    public CweCodeQl(String backendAddress) {
        super("CweCodeQl", null);
        this.backendAddress = backendAddress;
    }

    // Methods
    /**
     * @param projectLocation The path to a binary file for the desired solution of
     *                        project to analyze
     * @return The path to the analysis results file
     */
    @Override
    public Path analyze(Path projectLocation) {
        String projectName = projectLocation.getFileName().toString();
        LOGGER.info(this.getName() + "  Analyzing " + projectName);
        System.out.println("Analyzing " + projectName + " with " + this.getName());

        // set up results dir

        String workingDirectoryPrefix = "";
        String outputFilePath = "";
        
        try {
            // Load properties
            Properties prop = PiqueProperties.getProperties("src/main/resources/pique-properties.properties");
            Path resultsDir = Paths.get(prop.getProperty("results.directory"));

            // Set up working directory
            workingDirectoryPrefix = resultsDir + "/tool-out/CWE-200/";
            Files.createDirectories(Paths.get(workingDirectoryPrefix));
            // Set up output file path
            outputFilePath = workingDirectoryPrefix + "result.csv";
        
        } catch (java.io.IOException e) {
            e.printStackTrace();
            LOGGER.debug("Error creating directory to save CweQodeQl tool results");
            System.out.println("Error creating directory to save CweQodeQl tool results");
        }
        
        // Check if the results file already exists
        if (!doesExist(workingDirectoryPrefix, projectName)){
        
            // Check to see if the server is running
            if (isServerRunning(backendAddress)){
                LOGGER.info("Server is running at: " + backendAddress);
                // Upload the project to the server
                Path zipPath = zipProject(projectLocation);
                uploadProjectToServer(backendAddress, zipPath);
                
                // Perform the analysis
                String toolResults = sendPostRequestToServer(backendAddress, projectName);
                JSONObject jsonResponse = responseToJSON(toolResults);

                try {
                    if (jsonResponse.has("error") && !jsonResponse.isNull("error")) {
                        System.out.println("Error running CweQodeQl on " + projectName + " " + jsonResponse.getString("error"));
                        LOGGER.error("Error running CweQodeQl on " + projectName + " " + jsonResponse.getString("error"));
                        return null;
                    }
                } catch (JSONException e) {
                    return null;
                }
                
                // Convert the results to a CSV file, and save it
                Path finalResults = saveToolResultsToFile(jsonResponse, outputFilePath);
                
                parseAnalysis(finalResults);
                return finalResults;

            } else {
                // TODO: Start the server
                LOGGER.error("Server is not running at: " + backendAddress);
                return null;
            }
        }
        // Something went wrong 
        return null;
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
        System.out.println(this.getName() + " Parsing Analysis...");
        LOGGER.debug(this.getName() + " Parsing Analysis...");

        Map<String, Diagnostic> diagnostics = helperFunctions.initializeDiagnostics(this.getName());
        List<String[]> results = null;

        try {
            results = helperFunctions.readFileContent(toolResults);
            for (String[] record : results) {
                String cweId = record[0].split(":")[0];
                String cweDescription = record[0].split(":")[1].trim();
                String filePath = record[4];
                int lineNumber = Integer.parseInt(record[5]);
                int characterNumber = Integer.parseInt(record[6]);
                int severity = this.cweToSeverity(cweId);
               
                Diagnostic diag = diagnostics.get((cweId + " Diagnostic CweCodeQl"));
                if (diag != null) {
                    Finding finding = new Finding(filePath, 
                                            lineNumber, 
                                            characterNumber,
                                            severity);
                    finding.setName(cweDescription);
                    diag.setChild(finding);
                    System.out.println(cweId + " Diagnostic CweCodeQl");
                }
                
            }

        } catch (IOException e) {
            LOGGER.info("No results to read from CweQodeQl.");
        }
        return diagnostics;
    }

    @Override
    public Path initialize(Path toolRoot) {
        final String[] cmd = { "codeql", "version" };

        try {
            helperFunctions.getOutputFromProgram(cmd);
        } catch (Exception e) {
            e.printStackTrace();
            LOGGER.error("Failed to initialize " + this.getName());
            LOGGER.error(e.getStackTrace().toString());
        }

        return toolRoot;
    }
    
    private JSONObject responseToJSON(String response) {
        try {
            JSONObject jsonResponse = new JSONObject(response);
            return jsonResponse;
        } catch (JSONException e) {
            LOGGER.error("Failed to parse JSON response from server.");
            e.printStackTrace();
        }
        return null;
    }

    private int cweToSeverity(String cweId){
        switch (cweId) {
            case "CWE-201": return 9;
            case "CWE-208": return 7;
            case "CWE-214": return 9;
            case "CWE-215": return 7;
            case "CWE-531": return 7;
            case "CWE-532": return 9;
            case "CWE-535": return 7;
            case "CWE-536": return 9;
            case "CWE-537": return 7;
            case "CWE-538": return 9;
            case "CWE-540": return 9;
            case "CWE-548": return 7;
            case "CWE-550": return 9;
            case "CWE-598": return 9;
            case "CWE-615": return 7;
            default: return 4;
            
        }
    }

    private boolean isServerRunning(String backendAddress) {
        try {
            String[] cmd = { "curl", backendAddress };
            helperFunctions.getOutputFromProgram(cmd);
        } catch (Exception e) {
            LOGGER.error("Server is not running at: " + backendAddress);
            return false;
        }
        return true;
    }

    private Path zipProject(Path projectLocation) {
        // Validate that projectLocation exists and is a directory
        if (!Files.exists(projectLocation)) {
            LOGGER.error("The project location does not exist: {}", projectLocation.toAbsolutePath());
            return null;
        }
        if (!Files.isDirectory(projectLocation)) {
            LOGGER.error("The project location is not a directory: {}", projectLocation.toAbsolutePath());
            return null;
        }

        Path zipFilePath = Paths.get("zipped-repos/" + projectLocation.getFileName().toString());

        try {
            // Ensure the base directory for zipFilePath exists
            if (!Files.exists(zipFilePath.getParent())) {
                Files.createDirectories(zipFilePath.getParent());
            }

            // Zip the project directory
            try (ZipOutputStream zos = new ZipOutputStream(Files.newOutputStream(zipFilePath))) {
                Files.walkFileTree(projectLocation, new SimpleFileVisitor<Path>() {
                    @Override
                    public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
                        ZipEntry zipEntry = new ZipEntry(projectLocation.relativize(file).toString());
                        zos.putNextEntry(zipEntry);
                        Files.copy(file, zos);
                        zos.closeEntry();
                        return FileVisitResult.CONTINUE;
                    }

                    @Override
                    public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs) throws IOException {
                        if (!projectLocation.equals(dir)) { // Avoid adding the root directory itself as an entry
                            ZipEntry zipEntry = new ZipEntry(projectLocation.relativize(dir).toString() + "/");
                            zos.putNextEntry(zipEntry);
                            zos.closeEntry();
                        }
                        return FileVisitResult.CONTINUE;
                    }
                });
            }

            LOGGER.info("Project successfully zipped at {}", zipFilePath);
            return zipFilePath;  // Return the path to the zip file
        } catch (IOException e) {
            LOGGER.error("Failed to zip project.", e);
            return null;
        }
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
            helperFunctions.getOutputFromProgram(cmd);
        } catch (Exception e) {
            LOGGER.error("Failed to upload project to server at: " + backendAddress);
            e.printStackTrace();
        }
    }

    private String sendPostRequestToServer(String backendAddress, String projectName) {
        String toolResults = null;
        try {
            // Properly format the JSON data with extra escaping
            String jsonData = "\"{\\\"project\\\":\\\"" + projectName + 
            "\\\", \\\"extension\\\":\\\"csv\\\", \\\"format\\\":\\\"csv\\\"}\"";
            
            // Define the command array with properly escaped JSON
            String[] cmd = {
                "curl", 
                "-X", "POST",
                "-H", "Content-Type: application/json",
                "-d", jsonData,
                backendAddress + "/codeql/"
            };
            
            // Execute the command
            toolResults = helperFunctions.getOutputFromProgramAsString(cmd);
            
        } catch (Exception e) {
            LOGGER.error("Failed to send POST request to server at: " + backendAddress);
            e.printStackTrace();
        }
        return toolResults;
    }
    
    private Path saveToolResultsToFile(JSONObject jsonResponse, String outputFilePath) {
        try {
            // Parse JSON response to extract "data" field
            String data = jsonResponse.getString("data");
    
            // Clean up the CSV string if it has extra quotes at the beginning or end
            // data = data.replaceAll("^\"|\"$", ""); // Remove surrounding quotes if present
            Path outputPath = Paths.get(outputFilePath);
            // Write the extracted data to the specified file path
            Files.write(outputPath, data.getBytes());
            
            LOGGER.info("Tool results saved to file successfully: " + outputFilePath);
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
    
    private boolean doesExist(String workingDirectoryPrefix, String projectName) {
        File tempResults = new File(workingDirectoryPrefix + "CweQodeQl " + projectName + ".json");
        if (tempResults.exists()) {
            LOGGER.info("Already ran CweQodeQl on: " + projectName + ", results stored in: " + tempResults.toString());
            return true;
        }
        LOGGER.info("Have not run CweQodeQl on: " + projectName + ", running now and storing in:"
                    + tempResults.toString());
            tempResults.getParentFile().mkdirs();
        return false;
    }

}
