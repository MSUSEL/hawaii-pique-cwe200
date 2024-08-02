import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.ServletException;
import java.io.IOException;

public class BAD_SensitiveNullPointerExceptionHandling extends HttpServlet {

    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String accountKey = null;

        try {
            if (request.getParameter("secretKey") != null) {
                accountKey = request.getParameter("secretKey");
            } else {
                throw new NullPointerException("Secret key is missing in the request.");
            }
            // Process the account key
        } catch (NullPointerException e) {
            response.sendError(HttpServletResponse.SC_BAD_REQUEST, accountKey + " is null.");
        }
    }

    private static String getSecretKey() {
        // Simulate a failure in secret key retrieval
        return null;
    }
}
