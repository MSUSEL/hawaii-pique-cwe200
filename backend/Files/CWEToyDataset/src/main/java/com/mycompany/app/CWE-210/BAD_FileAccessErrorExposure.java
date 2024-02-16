import java.io.FileNotFoundException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class BAD_FileAccessErrorExposure {
    // Vulnerability: Generating error messages that include the file system structure or path details when file access fails.

    public static void readFile(String filePath) {
        try {
            Files.readAllLines(Paths.get(filePath));
        } catch (FileNotFoundException e) {
            // Insecure: Error message directly exposes attempted file path, revealing system structure or existence of specific files.
            System.err.println("File not found exception: " + e.getMessage());
        } catch (Exception e) {
            // Generic catch block for other exceptions
            System.err.println("An error occurred while reading the file: " + filePath);
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        readFile("/etc/secret-config.properties");
    }
}
