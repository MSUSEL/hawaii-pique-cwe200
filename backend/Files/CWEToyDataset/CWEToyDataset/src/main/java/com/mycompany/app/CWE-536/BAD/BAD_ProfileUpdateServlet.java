import java.io.IOException;
import javax.servlet.*;
import javax.servlet.http.*;

public class BAD_ProfileUpdateServlet extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String userId = request.getParameter("userId");
        String email = request.getParameter("email");
        updateEmail(userId, email);

        try{
            updateEmail(userId, email);
        } catch (ServletException e) {
            response.getWriter().println("Error updating email for user ID: " + userId);
        }
    
    }

    private void updateEmail(String userId, String email) throws ServletException {
        // Email update logic that might fail
        throw new ServletException();
    }
}
