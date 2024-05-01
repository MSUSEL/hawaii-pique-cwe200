import javax.servlet.*;
import javax.servlet.http.*;
import java.io.*;

public class BAD_UserDetailsExposureServlet extends HttpServlet {
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String userId = request.getParameter("userId");
        
        try {
            String profile = System.getenv("USERPROFILE");
            request.setAttribute("profile", profile);
            request.getRequestDispatcher("/profile.jsp").forward(request, response);
        } catch (Exception e) {
            response.setContentType("text/html");
            PrintWriter out = response.getWriter();
            // Exposing detailed user information in error messages
            out.println("Error retrieving profile for user ID: " + userId + " - Error: " + e.getMessage());
        }
    }

}
