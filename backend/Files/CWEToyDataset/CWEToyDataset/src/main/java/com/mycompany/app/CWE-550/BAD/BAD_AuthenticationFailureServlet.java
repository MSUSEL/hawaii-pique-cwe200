import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.ejb.EJBException;
import java.io.IOException;

public class BAD_AuthenticationFailureServlet extends HttpServlet {
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws IOException {
        String username = request.getParameter("username");
        String password = request.getParameter("password");
        try {
            authenticateUser(username, password);
        } catch (EJBException e) {
            response.getWriter().write("Authentication failed: " + e.getMessage());
        }
    }

    private void authenticateUser(String username, String password) throws EJBException {
        if (username == null || password == null) {
            throw new EJBException("Username or password is null. Provided username: " + username + ", password: " + password);
        }
        throw new EJBException("Invalid credentials provided for user " + username + ". Attempted password: " + password);
    }
}