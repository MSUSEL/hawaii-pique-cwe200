import java.io.IOException;

import javax.servlet.*;
import javax.servlet.http.*;

public class BAD_UserProfileServlet extends HttpServlet {
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String userId = request.getParameter("userId");
        fetchUserProfile(userId);
    }

    private void fetchUserProfile(String userId) throws ServletException {
        // Logic to retrieve user profile that might throw ServletException
        throw new ServletException("Database access failed for user ID: " + userId);
    }
}
