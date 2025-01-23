import javax.servlet.*;
import javax.servlet.http.*;
import java.io.*;
import org.apache.logging.log4j.*;

public class BAD_FirewallConfigurationServlet extends HttpServlet {

    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String firewallRule = request.getParameter("rule");
        String command = "/usr/bin/updateFirewall " + firewallRule;

        try {
            Process process = Runtime.getRuntime().exec(command);
            int exitCode = process.waitFor();

            if (exitCode != 0) {
                BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
                String errorDetails = errorReader.readLine();
                System.err.println("Failed to update firewall rules: " + errorDetails);
                response.getWriter().write("Failed to update firewall rules. Please contact system administrator.");
            }
        } catch (InterruptedException | IOException e) {
            System.err.println("Error updating firewall: " + e.getMessage());
            response.setStatus(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
        }
    }
}
