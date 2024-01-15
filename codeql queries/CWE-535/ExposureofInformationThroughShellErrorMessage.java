/* Description -
A command shell error message indicates that there exists an unhandled exception in the web application code.
In many cases, an attacker can leverage the conditions that cause these errors in order to gain unauthorized access to the system.

https://cwe.mitre.org/data/definitions/535.html
*/

// Kyler - Basically we don't want to write the error out becasue it will give addition information.
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class ExposureofInformationThroughShellErrorMessage {
    // This method attempts to execute a shell command with user input
    public static String insecureCommandExecution(String userInput) {
        try {
            Process process = Runtime.getRuntime().exec(userInput);
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            StringBuilder output = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
            }
            process.waitFor();
            return output.toString();
        } catch (IOException | InterruptedException e) {
            // Catching the exception and exposing the error message
            return "Error: " + e.getMessage();

        }
    }

}
