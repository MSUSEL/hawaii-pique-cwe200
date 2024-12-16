import javax.servlet.http.*;
import javax.servlet.*;
import java.io.*;
import java.nio.file.*;

// GOOD: Securely serves files without exposing directory listings.
public class GOOD_DocumentAccessServlet extends HttpServlet {
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String requestedFileName = request.getParameter("filename");

        // Validate the requested file name to prevent directory traversal attacks
        if (requestedFileName != null && requestedFileName.matches("[a-zA-Z0-9_\\-\\.]+")) {
            Path fileDirectory = Paths.get(getServletContext().getRealPath("/uploaded-documents"));
            Path filePath = fileDirectory.resolve(requestedFileName).normalize();

            // Ensure the requested file is within the safe directory and exists
            if (filePath.startsWith(fileDirectory) && Files.exists(filePath)) {
                response.setContentType(getServletContext().getMimeType(requestedFileName));
                Files.copy(filePath, response.getOutputStream());
            } else {
                // File does not exist or attempting directory traversal
                response.sendError(HttpServletResponse.SC_NOT_FOUND, "File not found or access denied.");
            }
        } else {
            // Invalid file request
            response.sendError(HttpServletResponse.SC_BAD_REQUEST, "Invalid or missing file name.");
        }
    }
}
