import javax.servlet.*;
import javax.servlet.http.*;
import java.io.IOException;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

public class BAD_DbConnectionErrorServlet extends HttpServlet {
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String dbUrl = request.getParameter("dbUrl");
        String dbUser = request.getParameter("dbUser");
        String dbPass = request.getParameter("dbPass");

        Connection conn = DriverManager.getConnection(dbUrl, dbUser, dbPass);
        // Perform database operations...
    }

}
