import javax.servlet.*;
import javax.servlet.http.*;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

public class GOOD_GenericErrorWithLoggingServlet extends HttpServlet {
    private static final Logger LOGGER = Logger.getLogger(GOOD_GenericErrorWithLoggingServlet.class.getName());

    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        try {
            // Sensitive operation that might fail
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "An error occurred", e);
            response.sendError(HttpServletResponse.SC_INTERNAL_SERVER_ERROR, "Internal server error. Please try again later.");
        }
    }
}
