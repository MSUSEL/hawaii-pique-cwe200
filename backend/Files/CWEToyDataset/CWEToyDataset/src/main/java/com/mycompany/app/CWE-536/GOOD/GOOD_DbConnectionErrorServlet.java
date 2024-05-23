import javax.servlet.*;
import javax.servlet.http.*;
import java.io.IOException;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

public class GOOD_DbConnectionErrorServlet extends HttpServlet {
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        try {
            Connection conn = DriverManager.getConnection("jdbc:mysql://localhost:3306/myapp", "user", "pass");
            // Perform database operations...
        } catch (SQLException e) {
            response.getWriter().write("Failed to connect to database.");
        }
    }

    @Override
    public void init() throws ServletException {
        super.init();
        try {
            Class.forName("com.mysql.jdbc.Driver");
        } catch (ClassNotFoundException e) {
            throw new ServletException("MySQL driver not found.");
        }
    }

    public void safeDoGet(HttpServletRequest request, HttpServletResponse response) {
        try {
            doGet(request, response);
        } catch (ServletException e) {
            try {
                response.getWriter().println("ServletException caught.");
            } catch (IOException ioException) {
                ioException.printStackTrace();
            }
        } catch (IOException e) {
            e.printStackTrace(); // Handle IOException separately if needed
        }
    } 
}
