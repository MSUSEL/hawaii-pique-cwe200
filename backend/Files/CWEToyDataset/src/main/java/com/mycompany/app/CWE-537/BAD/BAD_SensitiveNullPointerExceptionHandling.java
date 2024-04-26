import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.ServletException;
import java.io.IOException;
import java.io.PrintWriter;

public class BAD_SensitiveNullPointerExceptionHandling extends HttpServlet {

    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String secretKey = null;
        try {
            // Simulate retrieving a secret key from a method that returns null
            secretKey = getSecretKey();
            // Attempting to use the secret key which is null
            int length = secretKey.length(); // This will throw NullPointerException
        } catch (NullPointerException e) {
            // Inappropriately handling NullPointerException using HttpServletResponse
            response.sendError(HttpServletResponse.SC_BAD_REQUEST, "Failed to handle secret key properly: " + e.getMessage());
            // Exposing that the secret key is null through the response header
            response.addHeader("Error-Info", "Error due to null secret key: " + secretKey);
        }
    }

    private static String getSecretKey() {
        // Simulate a failure in secret key retrieval
        return null;
    }
}
