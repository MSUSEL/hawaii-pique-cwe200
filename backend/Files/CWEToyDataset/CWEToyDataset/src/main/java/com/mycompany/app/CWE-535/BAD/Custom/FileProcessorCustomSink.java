import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class FileProcessorCustomSink {

    public static void processFile(String filePath) {
        String command = "process_file --path " + filePath;

        try {
            Process process = Runtime.getRuntime().exec(command);
            BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            String error;

            while ((error = errorReader.readLine()) != null) {
                PrintErrorMessage.print("File processing error: " + error);
            }

            if (process.waitFor() != 0) {
                PrintErrorMessage.print("File processing failed, see logs for details.");
            }
        } catch (IOException | InterruptedException e) {
            PrintErrorMessage.print("Processing operation failed: " + e.getMessage());
        }
    }

    public static void main(String[] args) {
        processFile("/path/to/sensitive/file.txt");
    }
}
