import java.io.BufferedReader;
import java.io.InputStreamReader;

public class BAD_SensitiveInfoExposureViaShellError {

    public static void main(String[] args) {
        try {
            // The command includes sensitive information, such as a password or API key
            String[] command = {"/bin/sh", "-c", "somecommand --password=secret"};
            Process process = Runtime.getRuntime().exec(command);
            process.waitFor();

            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            String line = "";
            while ((line = reader.readLine()) != null) {
                // Printing error messages to the console or logging them can expose sensitive information
                System.out.println(line);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
