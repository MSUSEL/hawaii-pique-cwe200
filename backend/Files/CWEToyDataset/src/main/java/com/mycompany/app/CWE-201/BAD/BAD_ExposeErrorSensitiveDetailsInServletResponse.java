import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;

public class BAD_ExposeErrorSensitiveDetailsInServletResponse extends HttpServlet {
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws IOException {
        try {
            processRequest(request);
        } catch (Exception e) {
            // BAD: Exposing detailed error information directly in servlet response
            response.getWriter().println("Failed to process request: " + e.getMessage());
        }
    }

    private void processRequest(HttpServletRequest request) throws Exception {
        throw new Exception("Invalid request parameters");
    }
}
