import javax.servlet.*;
import javax.servlet.http.*;
import java.io.*;

public class BAD_CredentialExposureServlet extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String username = request.getParameter("username");
        String password = request.getParameter("password");
        
        try {
            authenticate(username, password);
        } catch (Exception e) {
            response.setContentType("text/html");
            PrintWriter out = response.getWriter();
            out.println("<html><body>");
            out.println("<h1>Authentication Failed</h1>");
            // Exposing user credentials directly in response
            out.println("Failed to authenticate user: " + username + " with password: " + password);
            out.println("</body></html>");
            // Also logging sensitive data
            System.err.println("Authentication error for user: " + username + " with password: " + password);
        }
    }

    private void authenticate(String username, String password) throws Exception {
        // Simulated authentication logic
        throw new Exception("Database connection error");
    }
}
