import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

public class BAD_DatabaseErrorServlet extends HttpServlet {
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws IOException {
        String password = request.getParameter("password");
        String email = request.getParameter("email");
        String url = request.getParameter("url");

        try {
            establishDatabaseConnection(email, password, url);
        } catch (SQLException e) {
            response.getWriter().write("Failed to connect to database: " + e.getMessage());
        }
    }

    private void establishDatabaseConnection(String email, String password, String url) throws SQLException {
        Connection conn = DriverManager.getConnection(url, email, password);
        // Additional database logic
        throw new SQLException("Could not establish connection to database using credentials: " + email + ", " + password + ", " + url);
    }
}
