import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;

public class BAD_VerboseLibraryExceptionHandling {
    public static void main(String[] args) {
        try {
            FileUtils.readFileToString(new File("/path/to/nonexistent/file"), "UTF-8");
        } catch (IOException e) {
            // Insecure practice: Exposing detailed file system path or library-specific error messages.
            e.printStackTrace();
        }
    }
}
