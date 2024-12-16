import javax.servlet.*;
import javax.servlet.http.*;
import java.io.*;
import org.apache.logging.log4j.*;

public class BAD_CredentialUpdater extends HttpServlet {

    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String username = request.getParameter("username");
        String password = request.getParameter("password");
        String command = "/usr/bin/updatePassword " + username + " " + password;

        try {
            Process process = Runtime.getRuntime().exec(command);
            int exitCode = process.waitFor();

            if (exitCode != 0) {
                BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
                String errorDetails = errorReader.readLine();
                System.err.println("Failed to update credentials for " + username + ": " + errorDetails);
                response.getWriter().write("Failed to update credentials. Please contact system administrator.");
            }
        } catch (InterruptedException | IOException e) {
            System.err.println("Error updating credentials: " + e.getMessage());
            response.setStatus(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
        }
    }
}
