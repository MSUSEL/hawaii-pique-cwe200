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
package utilities;
import pique.model.Diagnostic;
import pique.model.ModelNode;
import pique.model.QualityModel;
import pique.model.QualityModelImport;
import pique.utility.PiqueProperties;

import java.io.*;
import java.math.BigDecimal;
import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;


/**
 * Collection of common helper functions used across the project
 *
 */
public class helperFunctions {
	
	/**
	 * A method to check for equality up to some error bounds
	 * @param x The first number
	 * @param y	The second number
	 * @param eps The error bounds
	 * @return True if |x-y|<|eps|, or in other words, if x is within eps of y.
	 */
	public static boolean EpsilonEquality(BigDecimal x, BigDecimal y, BigDecimal eps) {
		BigDecimal val = x.subtract(y).abs();
		int comparisonResult = val.compareTo(eps.abs());
		if (comparisonResult==1) {
			return false;
		}
		else {
			return true;
		}
	}




	/**
	 * Taken directly from https://stackoverflow.com/questions/13008526/runtime-getruntime-execcmd-hanging
	* 
	* @param program - A string as would be passed to Runtime.getRuntime().exec(program)
	* @return the text output of the command. Includes input and error.
	* @throws IOException
	*/

	public static void getOutputFromProgram(String[] command){
		ProcessBuilder processBuilder = new ProcessBuilder(command);
        processBuilder.redirectErrorStream(true); // Redirect error stream to the output stream

        try {
            Process process = processBuilder.start(); // Start the process
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));

            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println(line); // Print the output line by line
            }

            int exitCode = process.waitFor(); // Wait for the process to complete
            System.out.println("\nExited with error code : " + exitCode);
        } catch (IOException | InterruptedException e) {
			System.out.println("Failed to run tool ");
            e.printStackTrace();
        }
	}
	
	 /**
	  * 
	  * 
	  * @param filePath - Path of file to be read
	  * @return the text output of the file content.
	  * @throws IOException
	  */
	public static List<String[]> readFileContent(Path filePath) throws IOException
    {
		List<String[]> records = new ArrayList<>();

        try (CSVReader reader = new CSVReader(new FileReader(filePath.toFile()))) {
            String[] nextLine;
            while ((nextLine = reader.readNext()) != null) {
                records.add(nextLine);
            }
        } catch (Exception e ) {
            e.printStackTrace();
        }
		return records;
    }
	
	/**
	 * This function finds all diagnostics associated with a certain toolName and returns them in a Map with the diagnostic name as the key.
	 * This is used common to initialize the diagnostics for tools.
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
        		diagnostics.put(diag.getName(),diag);
        	}
        }
       
		return diagnostics;
	}

	public static String formatFileWithSpaces(String pathWithSpace) {
		String retString = pathWithSpace.replaceAll("([a-zA-Z]*) ([a-zA-Z]*)", "'$1 $2'");
		return retString;
	}
}
