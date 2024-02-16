import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;

@WebServlet("/login")
public class BAD_LoginResponseDiscrepancy extends HttpServlet {
    @Override
    protected void doPost(HttpServletRequest req, HttpServletResponse resp) throws ServletException, IOException {
        String username = req.getParameter("username");
        String password = req.getParameter("password");

        if (isUserValid(username)) {
            if (isPasswordCorrect(username, password)) {
                resp.getWriter().write("Login successful");
            } else {
                resp.getWriter().write("Invalid password");
            }
        } else {
            resp.getWriter().write("User does not exist");
        }
    }

    private boolean isUserValid(String username) {
        // Placeholder for user validation logic
        return false;
    }

    private boolean isPasswordCorrect(String username, String password) {
        // Placeholder for password validation logic
        return false;
    }
}
