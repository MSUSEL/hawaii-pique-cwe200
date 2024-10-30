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
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Properties;

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
    private String openAiTokenPath;

    public CweCodeQl(String openAiTokenPath) {
        super("CweCodeQl", null);
        this.openAiTokenPath = openAiTokenPath;
    }

    // Methods
    /**
     * @param projectLocation The path to a binary file for the desired solution of
     *                        project to analyze
     * @return The path to the analysis results file
     */
    @Override
    public Path analyze(Path projectLocation) {
        String projectName = projectLocation.toString();
        LOGGER.info(this.getName() + "  Analyzing " + projectName);
        System.out.println("Analyzing " + projectName + " with " + this.getName());

        // set up results dir

        String workingDirectoryPrefix = "";
        String outpuFilePath = "";
        String queriesDir = "";
        try {
            Properties prop = PiqueProperties.getProperties("src/main/resources/pique-properties.properties");
            Path resultsDir = Paths.get(prop.getProperty("results.directory"));
            queriesDir = prop.getProperty("queries.directory");

            workingDirectoryPrefix = resultsDir + "/tool-out/CWE-200/";
            Files.createDirectories(Paths.get(workingDirectoryPrefix));
            outpuFilePath = workingDirectoryPrefix + "result.csv";
        } catch (java.io.IOException e) {
            e.printStackTrace();
            LOGGER.debug("Error creating directory to save CweQodeQl tool results");
            System.out.println("Error creating directory to save CweQodeQl tool results");
        }
        File tempResults = new File(workingDirectoryPrefix + "CweQodeQl " + projectName + ".json");
        if (tempResults.exists()) {
            LOGGER.info("Already ran CweQodeQl on: " + projectName + ", results stored in: " + tempResults.toString());
        } else {
            LOGGER.info("Have not run CweQodeQl on: " + projectName + ", running now and storing in:"
                    + tempResults.toString());
            tempResults.getParentFile().mkdirs();

            String[] createDatabaseCmd = {
                    "codeql", "database", "create", workingDirectoryPrefix + "projectDb",
                    "--language=java",
                    "--source-root=" + projectName
            };
            LOGGER.info(Arrays.toString(createDatabaseCmd));

            try {
                //helperFunctions.getOutputFromProgram(createDatabaseCmd);
            } catch (Exception e) {
                LOGGER.error("Failed to Create CodeQl Database");
                LOGGER.error(e.toString());
                e.printStackTrace();
            }

            String[] analyzeDatabaseCmd = {
                    "codeql", "database",
                    "analyze", workingDirectoryPrefix + "projectDb",
                    "--format=csv",
                    "-o=" + outpuFilePath, queriesDir
            };

            LOGGER.info(Arrays.toString(analyzeDatabaseCmd));
            try {
                //helperFunctions.getOutputFromProgram(analyzeDatabaseCmd);
            } catch (Exception e) {
                LOGGER.error("Failed to analyze CodeQl Databse");
                LOGGER.error(e.toString());
                e.printStackTrace();
            }

        }

        File finalResults = new File(outpuFilePath);
        return finalResults.toPath();
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
               
                Diagnostic diag = diagnostics.get((cweId + " Diagnostic CweCodeQl"));
                if (diag != null) {
                    int severity=this.severityToInt("high");
                    Finding finding = new Finding(record[4], 
                                            Integer.parseInt(record[5]), 
                                            Integer.parseInt(record[6]),
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

    // maps low-critical to numeric values based on the highest value for each
    // range.
    private Integer severityToInt(String severity) {
        Integer severityInt = 1;
        switch (severity.toLowerCase()) {
            case "low": {
                severityInt = 4;
                break;
            }
            case "medium": {
                severityInt = 7;
                break;
            }
            case "high": {
                severityInt = 9;
                break;
            }
            case "critical": {
                severityInt = 10;
                break;
            }
        }

        return severityInt;
    }

}
