import java.io.IOException;

import javax.servlet.*;
import javax.servlet.http.*;

public class BAD_UserProfileServlet extends HttpServlet {
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String userId = request.getParameter("userId");
        String passPhrase = request.getParameter("passPhrase");
        try{
            fetchUserProfile(userId, passPhrase);
        } catch (ServletException e) {
            response.getWriter().println(e.getMessage());
        }
    }

    private void fetchUserProfile(String userId, String PassPhrase) throws ServletException {
        // Logic to retrieve user profile that might throw ServletException
        throw new ServletException("User profile retrieval failed for user: " + userId + " with passphrase: " + PassPhrase);
    }
}
