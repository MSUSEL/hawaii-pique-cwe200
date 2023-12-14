import java.io.IOException;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

@WebServlet("/HandlerServlet")
public class HandlerServlet extends HttpServlet {

    protected void doGet(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        response.setContentType("text/html");

        try (Connection connection = DriverManager.getConnection("jdbc:yourDatabaseUrl", "username", "password");
             PreparedStatement statement = connection.prepareStatement("SELECT name FROM users");
             ResultSet resultSet = statement.executeQuery()) {
            
            StringBuilder users = new StringBuilder();
            while (resultSet.next()) {
                users.append(resultSet.getString("name")).append("<br>");
            }

            response.getWriter().write("User names:<br>" + users.toString());
        } catch (SQLException ex) {
            response.getWriter().write("Database error: " + ex.getMessage());
        }
    }
}
