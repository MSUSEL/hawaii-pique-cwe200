import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.ServletException;
import java.io.IOException;

@WebServlet("/fetchData")
public class GOOD_FetchPublicDataServlet extends HttpServlet {
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String category = request.getParameter("category");
        
        // Logic to fetch data based on the category. Assume it's public information like articles or product listings.
        // The category itself is not sensitive information.
        
        response.setContentType("application/json");
        response.getWriter().write("{\"data\":\"Sample data for category: " + category + "\"}");
    }
}
