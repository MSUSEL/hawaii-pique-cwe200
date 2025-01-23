import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.ejb.EJBException;
import java.io.IOException;

public class BAD_AuthenticationFailureServlet extends HttpServlet {
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws IOException {
        try {
            authenticateUser(request.getParameter("username"), request.getParameter("password"));
        } catch (EJBException e) {
            response.getWriter().write("Authentication failed: " + e.getMessage());
            response.setStatus(HttpServletResponse.SC_UNAUTHORIZED);
        }
    }

    private void authenticateUser(String username, String password) throws EJBException {
        if (username == null || password == null) {
            throw new EJBException("Username or password is null");
        }
        // Authentication logic that might fail
        throw new EJBException("Invalid credentials provided for user " + username);
    }
}
