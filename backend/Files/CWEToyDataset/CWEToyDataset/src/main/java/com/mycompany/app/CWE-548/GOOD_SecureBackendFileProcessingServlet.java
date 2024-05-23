import javax.servlet.http.*;
import javax.servlet.*;
import java.io.*;
import java.util.Date;

// GOOD: Securely processes files in a directory for backend operations without exposing file listings to users.
public class GOOD_SecureBackendFileProcessingServlet extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        // Path to a directory containing temporary files
        String tempFilesPath = getServletContext().getRealPath("/temp-files");
        File tempDirectory = new File(tempFilesPath);

        // Verify the directory exists and is a directory
        if (!tempDirectory.exists() || !tempDirectory.isDirectory()) {
            System.err.println("Temporary files directory does not exist or is not a directory.");
            return;
        }

        // Processing files securely without exposing listings to users
        for (File file : tempDirectory.listFiles()) {
            if (file.isFile() && isOlderThanOneDay(file.lastModified())) {
                boolean deleted = file.delete();
                if (deleted) {
                    System.out.println("Deleted old temporary file.");
                } else {
                    System.err.println("Failed to delete temporary file. ");
                }
            }
        }
    }

    private boolean isOlderThanOneDay(long fileTimestamp) {
        // Check if the file's last modified timestamp is older than one day
        final long oneDayInMillis = 24 * 60 * 60 * 1000;
        long age = new Date().getTime() - fileTimestamp;
        return age > oneDayInMillis;
    }
}
