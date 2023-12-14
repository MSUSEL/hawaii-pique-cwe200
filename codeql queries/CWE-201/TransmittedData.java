import java.io.IOException;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

@WebServlet("/LoginServlet")
public class LoginServlet extends HttpServlet {

    private static final String CORRECT_PASSWORD = "mySecretPassword123";

    protected void doPost(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        String username = request.getParameter("username");
        String password = request.getParameter("password");

        if (password != null && password.equals(CORRECT_PASSWORD)) {
            response.getWriter().write("Welcome, " + username + "!");
        } else {
            response.getWriter().write("Authentication failed. Invalid password: " + password);
        }
    }
}