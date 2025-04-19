import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;

public class BAD_DetailedRuntimeExceptionServlet extends HttpServlet {
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws IOException {
        String userId = request.getParameter("userId");
        try {
            String username  = fetchUsername(userId);
            String email     = fetchEmail(userId);
            String ssn       = fetchSSN(userId);
            String lastLogin = fetchLastLogin(userId);
            throw new RuntimeException(
                "Failed to process user with ID "   + userId   +
                ", username="                         + username +
                ", email="                            + email    +
                ", SSN="                              + ssn      +
                ", lastLogin="                        + lastLogin
            );
        } catch (RuntimeException e) {
            response.sendError(
                HttpServletResponse.SC_INTERNAL_SERVER_ERROR,
                "An error occurred: " + e.getMessage()
            );
        }
    }

    private String fetchUsername(String userId) {
        return "jdoe"; 
    }

    private String fetchEmail(String userId) {
        return "jdoe@example.com";
    }

    private String fetchSSN(String userId) {
        return "123-45-6789";
    }

    private String fetchLastLogin(String userId) {
        return "2025-04-10T15:30:00Z";
    }
}