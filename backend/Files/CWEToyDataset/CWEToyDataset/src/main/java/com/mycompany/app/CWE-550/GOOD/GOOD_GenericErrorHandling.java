import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;

public class GOOD_GenericErrorHandling extends HttpServlet {
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws IOException {
        try {
            // Some sensitive operation
            throw new Exception("Sensitive error message");
        } catch (Exception e) {
            // Secure: Sending a generic error message to the client
            response.sendError(HttpServletResponse.SC_INTERNAL_SERVER_ERROR, "An internal server error occurred. Please try again later.");
        }
    }
}
