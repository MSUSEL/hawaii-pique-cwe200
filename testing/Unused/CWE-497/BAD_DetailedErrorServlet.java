import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;

@WebServlet("/errorDetails")
public class BAD_DetailedErrorServlet extends HttpServlet {
    @Override
    protected void doGet(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        try {
            processRequest(req);
        } catch (Exception e) {
            // Insecure: Exposing stack trace and system details in the HTTP response
            e.printStackTrace(resp.getWriter()); // Never do this in a real application!
            resp.setContentType("text/plain");
            resp.setStatus(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
            resp.getWriter().write("An error occurred: " + e.getMessage());
        }
    }

    private void processRequest(HttpServletRequest req) throws Exception {
        // Simulated processing logic that may throw exceptions
        throw new Exception("Failed to load configuration from /etc/app/config.properties");
    }
}
