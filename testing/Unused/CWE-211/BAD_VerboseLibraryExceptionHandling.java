import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.util.Map;

public class BAD_VerboseLibraryExceptionHandling {
    public static void main(String[] args) {
        try {
            // Simulating an attempt to read a critical configuration file
            FileUtils.readFileToString(new File("/secure/config/api_keys.config"), "UTF-8");
        } catch (IOException e) {
            // Insecure practice: Exposing detailed file system path or library-specific error messages
            // along with sensitive system environment information.
            System.err.println("Failed to load critical configuration: " + e.getMessage());
            
            // Directly printing the stack trace to standard error.
            e.printStackTrace();
        }
    }
}
