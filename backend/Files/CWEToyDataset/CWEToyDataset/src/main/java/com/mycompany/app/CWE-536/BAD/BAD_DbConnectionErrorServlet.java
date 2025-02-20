import jakarta.servlet.*;
import jakarta.servlet.http.*;

import java.io.IOException;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

public class BAD_DbConnectionErrorServlet extends HttpServlet {
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        response.setContentType("text/html");

        String dbUrl = request.getParameter("dbUrl");
        String dbUser = request.getParameter("dbUser");
        String dbPass = request.getParameter("dbPass");
        Connection conn = null;

        try {
            if (dbUrl == null || dbUser == null || dbPass == null) {
                throw new ServletException("Invalid database connection parameters: " + dbUrl + ", " + dbUser + ", " + dbPass);
            }

            // Establish a database connection
            conn = DriverManager.getConnection(dbUrl, dbUser, dbPass);
            response.getWriter().println("<p>Database connection successful.</p>");

        } catch (SQLException e) {
            System.err.println("Failed to connect to the database.");
            response.sendError(HttpServletResponse.SC_INTERNAL_SERVER_ERROR, "Database connection failed.");
        } catch (ServletException e) {
            System.err.println("Servlet Exception: " + e.getMessage());
            response.sendError(HttpServletResponse.SC_BAD_REQUEST, e.getMessage());
        } finally {
            try {
                if (conn != null) conn.close();
            } catch (SQLException e) {
                System.err.println("Error closing the connection: " + e.getMessage());
            }
        }
    }
}
