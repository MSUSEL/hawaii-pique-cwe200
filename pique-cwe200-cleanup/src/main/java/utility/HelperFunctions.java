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
package utility;

import pique.model.Diagnostic;
import pique.model.ModelNode;
import pique.model.QualityModel;
import pique.model.QualityModelImport;
import pique.utility.PiqueProperties;

import java.io.*;
import java.math.BigDecimal;

import com.opencsv.CSVParserBuilder;
import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;
import com.opencsv.exceptions.CsvException;

import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.*;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Collection of common helper functions used across the project
 *
 */
public class HelperFunctions {
    private static final Logger LOGGER = LoggerFactory.getLogger(HelperFunctions.class);

    /**
     * A method to check for equality up to some error bounds
     * 
     * @param x   The first number
     * @param y   The second number
     * @param eps The error bounds
     * @return True if |x-y|<|eps|, or in other words, if x is within eps of y.
     */
    public static boolean EpsilonEquality(BigDecimal x, BigDecimal y, BigDecimal eps) {
        BigDecimal val = x.subtract(y).abs();
        int comparisonResult = val.compareTo(eps.abs());
        if (comparisonResult == 1) {
            return false;
        } else {
            return true;
        }
    }

    /**
     * Taken directly from
     * https://stackoverflow.com/questions/13008526/runtime-getruntime-execcmd-hanging
     *
     * @param command - A string as would be passed to
     *                Runtime.getRuntime().exec(program)
     * @return the text output of the command. Includes input and error.
     * @throws IOException
     */

    public static void getOutputFromProgram(String[] command) {
        ProcessBuilder processBuilder = new ProcessBuilder(command);
        processBuilder.redirectErrorStream(true); // Redirect error stream to the output stream

        try {
            Process process = processBuilder.start(); // Start the process
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));

            // String line;
            while ((reader.readLine()) != null) {
                // System.out.println(line); // Print the output line by line
            }
            process.waitFor(); // Wait for the process to complete
            // System.out.println("\nExited with error code : " + exitCode);
        }

        catch (IOException | InterruptedException e) {
            // System.out.println("Failed to run tool ");
            // e.printStackTrace();
        }
    }

    public static String getOutputFromProgramAsString(String[] command) {
        ProcessBuilder processBuilder = new ProcessBuilder(command);
        processBuilder.redirectErrorStream(true); // Redirect error stream to the output stream

        StringBuilder jsonOutput = new StringBuilder();
        boolean jsonStarted = false; // Flag to track if JSON has started

        try {
            Process process = processBuilder.start(); // Start the process
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));

            String line;
            while ((line = reader.readLine()) != null) {
                // Check if JSON output has started and skip progress lines
                if (!jsonStarted) {
                    if (line.trim().contains("{")) {
                        jsonStarted = true;
                    } else {
                        continue; // Skip progress lines
                    }
                }
                // Accumulate JSON lines
                jsonOutput.append(line);
            }

            int exitCode = process.waitFor(); // Wait for the process to complete
            System.out.println("\nExited with error code : " + exitCode);

        } catch (IOException | InterruptedException e) {
            System.out.println("Failed to run tool ");
            e.printStackTrace();
        }

        return jsonOutput.toString(); // Return only the JSON data
    }

    /**
     *
     *
     * @param filePath - Path of file to be read
     * @return the text output of the file content.
     * @throws IOException
     */
    public static List<String[]> readFileContent(Path filePath) throws IOException {
        List<String[]> records = new ArrayList<>();

        try (CSVReader reader = new CSVReaderBuilder(new FileReader(filePath.toFile()))
                .withCSVParser(new CSVParserBuilder()
                        .withSeparator(',') // Set comma as the delimiter
                        .withQuoteChar('"') // Handle quoted strings
                        .build())
                .build()) {
            String[] nextLine;
            while ((nextLine = reader.readNext()) != null) {
                // System.out.println(String.join(",", nextLine)); // Print for debugging
                records.add(nextLine);
            }
        } catch (Exception e) {
            // e.printStackTrace();
        }
        return records;
    }

    /**
     * This function finds all diagnostics associated with a certain toolName and
     * returns them in a Map with the diagnostic name as the key.
     * This is used common to initialize the diagnostics for tools.
     * 
     * @param toolName The desired tool name
     * @return All diagnostics in the model structure with tool equal to toolName
     */
    public static Map<String, Diagnostic> initializeDiagnostics(String toolName) {
        // load the qm structure
        Properties prop = PiqueProperties.getProperties();
        Path blankqmFilePath = Paths.get(prop.getProperty("blankqm.filepath"));
        QualityModelImport qmImport = new QualityModelImport(blankqmFilePath);
        QualityModel qmDescription = qmImport.importQualityModel();

        Map<String, Diagnostic> diagnostics = new HashMap<>();

        // for each diagnostic in the model, if it is associated with this tool,
        // add it to the list of diagnostics
        for (ModelNode x : qmDescription.getDiagnostics().values()) {

            Diagnostic diag = (Diagnostic) x;
            if (diag.getToolName().equals(toolName)) {
                diagnostics.put(diag.getName(), diag);
            }
        }

        return diagnostics;
    }

    public static String formatFileWithSpaces(String pathWithSpace) {
        String retString = pathWithSpace.replaceAll("([a-zA-Z]*) ([a-zA-Z]*)", "'$1 $2'");
        return retString;
    }

    public static JSONObject responseToJSON(String response) {
        if (response == null || response.trim().isEmpty()) {
            LOGGER.error("Response is null or empty, cannot parse to JSON.");
            return null;
        }

        try {
            // Use regex to extract the JSON part
            Pattern jsonPattern = Pattern.compile("\\{.*");
            Matcher matcher = jsonPattern.matcher(response);

            if (matcher.find()) {
                String jsonPart = matcher.group(); // Extract the JSON substring
                return new JSONObject(jsonPart); // Parse it into a JSONObject
            } else {
                LOGGER.error("No JSON found in response: {}", response);
            }
        } catch (JSONException e) {
            LOGGER.error("Failed to parse JSON response: {}", response, e);
        }

        return null;
    }

    /**
     * This function takes a project name and a path to a project info file and
     * returns the java version for that project.
     * 
     * @param projectInfoFilePath The path to the project info file
     * @param projectName         The name of the project
     * @return The java version for the project
     */

    public static String getJavaVersion(Path projectInfoFilePath, String projectName) {
        try {
            // Read the entire JSON file as a UTF-8 string.
            String jsonString = new String(Files.readAllBytes(projectInfoFilePath), StandardCharsets.UTF_8);

            // Parse the JSON content.
            JSONObject jsonObject = new JSONObject(jsonString);
            JSONArray projects = jsonObject.getJSONArray("projects");

            // Loop through the projects array.
            for (int i = 0; i < projects.length(); i++) {
                JSONObject project = projects.getJSONObject(i);
                if (project.getString("projectName").equals(projectName)) {
                    // Get the javaVersion as a string, then parse it into an int.
                    LOGGER.info("Java version " + project.getString("javaVersion"));
                    return project.getString("javaVersion");
                }
            }
        } catch (IOException e) {
            // Handle errors reading the file.
            e.printStackTrace();
        } catch (JSONException e) {
            // Handle errors during JSON parsing.
            e.printStackTrace();
        } catch (NumberFormatException e) {
            // Handle any errors converting the javaVersion to int.
            e.printStackTrace();
        }

        // Return Java 11 as the default version if not found.
        LOGGER.info("Java version not found for " + projectName + ", defaulting to 11.");
        return "11";
    }
}
