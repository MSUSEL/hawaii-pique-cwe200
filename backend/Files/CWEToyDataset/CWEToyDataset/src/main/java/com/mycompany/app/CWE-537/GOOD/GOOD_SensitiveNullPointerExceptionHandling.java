import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.ServletException;
import java.io.IOException;
import java.io.PrintWriter;

public class GOOD_SensitiveNullPointerExceptionHandling extends HttpServlet {

    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String creditCardPin = null;
        try {
            // Simulate retrieving a secret key from a method that returns null
            creditCardPin = getSecretKey();
            // Attempting to use the secret key which is null
            int length = creditCardPin.length(); // This will throw NullPointerException
        } catch (NullPointerException e) {
            response.sendError(HttpServletResponse.SC_BAD_REQUEST, "Failed to handle secret key properly.");
        }
    }

    private static String getSecretKey() {
        // Simulate a failure in secret key retrieval
        return null;
    }
}
