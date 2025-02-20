import java.io.IOException;
import javax.servlet.*;
import javax.servlet.http.*;

public class BAD_ProfileUpdateServlet extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String oldEmail = request.getParameter("oldEmail");
        String newEmail = request.getParameter("newEmail");
        String pass = request.getParameter("password");

        updateEmail(oldEmail, newEmail, pass);

        try{
            updateEmail(oldEmail, newEmail, pass);
        } catch (ServletException e) {
            response.getWriter().println(e.getMessage());
        }
    
    }

    private void updateEmail(String oldEmail, String newEmail, String pass) throws ServletException {
        // Email update logic that might fail
        throw new ServletException("Email update failed for old email: " + oldEmail + " to new email: " + newEmail + " with password: " + pass);
    }
}
