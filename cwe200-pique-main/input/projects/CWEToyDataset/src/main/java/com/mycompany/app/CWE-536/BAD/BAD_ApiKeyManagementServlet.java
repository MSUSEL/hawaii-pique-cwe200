import java.io.IOException;
import javax.servlet.*;
import javax.servlet.http.*;
import org.apache.logging.log4j.*;

public class BAD_ApiKeyManagementServlet extends HttpServlet {
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String apiKey = request.getParameter("apiKey");
        try{
            if (apiKey == "apiKey" || apiKey.isEmpty()) {
                throw new ServletException("Invalid API key " + apiKey + " provided.");
            }
        }
        catch(ServletException e){
            System.err.println("Invalid API key " + apiKey + " provided.");
            response.sendError(HttpServletResponse.SC_INTERNAL_SERVER_ERROR, "An error occurred while processing the request: " + e.getMessage());
        }
    }

}