import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class BAD_FileReadSensitiveError {
    public static void main(String[] args) {
        final String filePath = "/etc/sensitive/config.txt";
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            // Read file...
        } catch (IOException e) {
            // Insecure: Exposing file path in the error message
            e.printStackTrace();
        }
    }
}
