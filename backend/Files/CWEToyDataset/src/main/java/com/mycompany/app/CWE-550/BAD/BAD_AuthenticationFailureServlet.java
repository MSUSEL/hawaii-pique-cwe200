import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;

public class BAD_AuthenticationFailureServlet extends HttpServlet {
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws IOException {
        try {
            authenticateUser(request.getParameter("username"), request.getParameter("password"));
        } catch (Exception e) {
            // Insecure: Sending authentication failure details back to the client
            response.getWriter().write("Authentication failed: " + e.getMessage());
            response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
        }
    }

    private void authenticateUser(String username, String password) throws Exception {
        if (username == null || password == null) {
            throw new Exception("Username or password is null");
        }
        // Authentication logic that might fail
        throw new Exception("Invalid credentials provided for user " + username);
    }
}
