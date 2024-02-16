import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;

public class GOOD_SecureLibraryExceptionHandling {
    public static void main(String[] args) {
        try {
            FileUtils.readFileToString(new File("/secure/path/valid/file"), "UTF-8");
        } catch (IOException e) {
            // Secure practice: Capture the exception and provide a generic error message.
            System.out.println("An error occurred processing your request. Please contact support.");
            // Optionally log the error internally without exposing details to the user.
            logError(e.getMessage());
        }
    }

    private static void logError(String errorDetails) {
        // Placeholder for logging mechanism. In a real scenario, this would securely log internally.
    }
}
