import java.io.IOException;

import javax.servlet.*;
import javax.servlet.http.*;

public class BAD_UserProfileServlet extends HttpServlet {
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String userId = request.getParameter("userId");
        try {
            fetchUserProfile(userId);
        } catch (ServletException e) {
            // Exposing user ID in the error message
            response.sendError(HttpServletResponse.SC_INTERNAL_SERVER_ERROR, "Error fetching profile for user ID: " + userId);
        }
    }

    private void fetchUserProfile(String userId) throws ServletException {
        // Logic to retrieve user profile that might throw ServletException
        throw new ServletException("Database access failed for user ID: " + userId);
    }
}
