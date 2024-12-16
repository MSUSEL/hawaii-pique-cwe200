import javax.servlet.*;
import javax.servlet.http.*;
import java.io.*;

public class GOOD_SecureFileAccessServlet extends HttpServlet {
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String filename = request.getParameter("filename");
        if (!isValidFilename(filename)) {
            response.sendError(HttpServletResponse.SC_BAD_REQUEST, "Invalid file request.");
            return;
        }

        File file = new File("/secure/data/" + filename);
        if (!file.exists()) {
            response.sendError(HttpServletResponse.SC_NOT_FOUND, "File not found.");
            return;
        }

    }

    private boolean isValidFilename(String filename) {
        // Implement validation logic to prevent directory traversal, etc.
        return filename != null && filename.matches("[\\w,\\s-]+\\.[A-Za-z]{3}");
    }
}
