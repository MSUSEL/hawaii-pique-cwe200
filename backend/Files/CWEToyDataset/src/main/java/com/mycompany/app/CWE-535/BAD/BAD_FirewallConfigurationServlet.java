import javax.servlet.*;
import javax.servlet.http.*;
import java.io.*;
import org.apache.logging.log4j.*;

public class BAD_FirewallConfigurationServlet extends HttpServlet {
    private static final Logger logger = LogManager.getLogger(BAD_FirewallConfigurationServlet.class);

    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String firewallRule = request.getParameter("rule");
        String command = "/usr/bin/updateFirewall " + firewallRule;

        try {
            Process process = Runtime.getRuntime().exec(command);
            int exitCode = process.waitFor();

            if (exitCode != 0) {
                BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
                String errorDetails = errorReader.readLine();
                logger.error("Failed to update firewall rules: {}", errorDetails); // Logging sensitive error details
                response.getWriter().write("Failed to update firewall rules. Please contact system administrator.");
            }
        } catch (InterruptedException | IOException e) {
            logger.error("Exception when updating firewall: {}", e.getMessage());
            response.setStatus(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
        }
    }
}
