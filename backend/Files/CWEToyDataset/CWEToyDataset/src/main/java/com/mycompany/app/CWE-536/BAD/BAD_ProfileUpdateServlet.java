import java.io.IOException;
import javax.servlet.*;
import javax.servlet.http.*;

public class BAD_ProfileUpdateServlet extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String userId = request.getParameter("userId");
        String email = request.getParameter("email"); // Potentially sensitive
        try {
            updateEmail(userId, email);
        } catch (ServletException e) {
            // Exposing potentially sensitive information indirectly via response
            response.getWriter().println("Failed to update email for user. Please retry or check system logs.");
            getServletContext().log("Update failed for user ID: " + userId + " with email: " + email, e);
        }
    }

    private void updateEmail(String userId, String email) throws ServletException {
        // Email update logic that might fail
        throw new ServletException("Email format error or email already in use: " + email);
    }
}
