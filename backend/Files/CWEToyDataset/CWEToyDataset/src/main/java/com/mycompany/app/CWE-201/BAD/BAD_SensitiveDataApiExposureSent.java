import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import org.json.JSONObject;

public class BAD_SensitiveDataApiExposureSent extends HttpServlet {
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws IOException {
        String accountNumber = getSensitiveAccountDetails(request.getParameter("userId"));

        // Prepare JSON response
        JSONObject jsonResponse = new JSONObject();
        jsonResponse.put("status", "success");
        jsonResponse.put("data", accountNumber);

        // Set response type and encoding
        response.setContentType("application/json");
        response.setCharacterEncoding("UTF-8");

        // Vulnerable: Sending sensitive data directly to the client via API response
        response.getWriter().write(jsonResponse.toString());
    }

    // Simulates retrieval of sensitive account details, e.g., bank account number
    private String getSensitiveAccountDetails(String userId) {
        // Example sensitive data retrieval
        return "Bank Account Number: 1234567890 for User ID: " + userId;
    }
}