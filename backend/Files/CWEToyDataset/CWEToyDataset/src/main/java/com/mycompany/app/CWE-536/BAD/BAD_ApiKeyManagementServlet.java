import java.io.IOException;
import javax.servlet.*;
import javax.servlet.http.*;
import org.apache.logging.log4j.*;

public class BAD_ApiKeyManagementServlet extends HttpServlet {
    private static final Logger logger = LogManager.getLogger(BAD_ApiKeyManagementServlet.class);

    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String apiKey = request.getParameter("apiKey"); // API key is received from the request, simulating a query parameter
        try {
            validateApiKey(apiKey);
        } catch (ServletException e) {
            // Logging the API key along with error details using Log4J, which is a bad practice as it exposes sensitive data
            logger.error("Failed to validate API key: " + apiKey, e);
            response.getWriter().println("API key validation failed. Please contact support.");
        }
    }

    private void validateApiKey(String apiKey) throws ServletException {
        // Simulate API key validation logic that might fail
        if (apiKey == null || apiKey.isEmpty()) {
            throw new ServletException("Invalid API key provided.");
        }
        // More validation logic could go here
    }
}