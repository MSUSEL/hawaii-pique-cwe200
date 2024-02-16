import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;

@WebServlet("/safeError")
public class GOOD_SafeErrorHandlingServlet extends HttpServlet {
    @Override
    protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        try {
            processRequest(req);
        } catch (Exception e) {
            // Secure: Log the error internally and provide a generic response
            logError("A processing error occurred, refer to application logs for details.", e);
            resp.setContentType("text/plain");
            resp.setStatus(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
            resp.getWriter().write("Internal server error. Please contact support.");
        }
    }

    private void processRequest(HttpServletRequest req) {
        // Request processing logic
    }

    private void logError(String message, Exception e) {
        // Internal logging mechanism
    }
}
