import javax.servlet.*;
import javax.servlet.http.*;
import java.io.*;
import org.apache.logging.log4j.*;

public class GOOD_CredentialUpdater extends HttpServlet {
    private static final Logger logger = LogManager.getLogger(GOOD_CredentialUpdater.class);

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
                logger.error("Failed to update credentials."); // Logging the error with username and details
                response.getWriter().write("Failed to update credentials. Please contact system administrator.");
            }
        } catch (InterruptedException | IOException e) {
            logger.error("Exception when updating credentials.");
            response.setStatus(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
        }
    }
}
