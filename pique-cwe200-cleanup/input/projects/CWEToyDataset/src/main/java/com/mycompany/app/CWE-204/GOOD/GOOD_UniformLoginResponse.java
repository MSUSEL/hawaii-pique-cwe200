import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;


@WebServlet("/secure-login")
public class GOOD_UniformLoginResponse extends HttpServlet {
    @Override
    protected void doPost(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        String username = req.getParameter("username");
        String password = req.getParameter("password");

        if (authenticate(username, password)) {
            resp.getWriter().write("Login successful");
        } else {
            resp.getWriter().write("Login failed. Please check your username and password.");
        }
    }

    private boolean authenticate(String username, String password) {
        return false;
    }
}
