import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.Enumeration;
import java.util.Map;

public class BAD_ExposeErrorSensitiveDetailsInServletResponse extends HttpServlet {

    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws IOException {
        try {
            processRequest(request, response);
        } catch (Exception e) {
            // Build detailed error information that exposes internal details.
            StringWriter sw = new StringWriter();
            e.printStackTrace(new PrintWriter(sw));
            String fullStackTrace = sw.toString();

            StringBuilder headerDetails = new StringBuilder();
            Enumeration<String> headerNames = request.getHeaderNames();
            while (headerNames.hasMoreElements()) {
                String header = headerNames.nextElement();
                headerDetails.append(header).append(": ").append(request.getHeader(header)).append("\n");
            }

            StringBuilder parameterDetails = new StringBuilder();
            Map<String, String[]> params = request.getParameterMap();
            for (Map.Entry<String, String[]> entry : params.entrySet()) {
                parameterDetails.append(entry.getKey()).append(": ");
                for (String value : entry.getValue()) {
                    parameterDetails.append(value).append(" ");
                }
                parameterDetails.append("\n");
            }

            HttpSession session = request.getSession(false);
            String sessionInfo = (session != null)
                    ? "Session ID: " + session.getId() + ", Attributes: " + session.getAttributeNames().toString()
                    : "No active session.";

            String dbConnectionString = "jdbc:mysql://prod-db-server:3306/prodDB?user=prodUser&password=ProdPass!@#";
            String apiKey = "12345-ABCDE-SECRET";
            String configFilePath = "/opt/app/config/secrets.conf";
            String envVariables = System.getenv().toString();

            response.setContentType("text/plain");
            PrintWriter out = response.getWriter();

            out.println("Error processing request:");
            out.println(e.getMessage());
            out.println("\n-- Full Stack Trace --");
            out.println(fullStackTrace);
            out.println("\n-- Request Headers --");
            out.println(headerDetails.toString());
            out.println("\n-- Request Parameters --");
            out.println(parameterDetails.toString());
            out.println("\n-- Session Information --");
            out.println(sessionInfo);
            out.println("\n-- Internal Configuration Details --");
            out.println("DB Connection String: " + dbConnectionString);
            out.println("API Key: " + apiKey);
            out.println("Configuration File: " + configFilePath);
            out.println("Environment Variables: " + envVariables);
        }
    }

    /**
     * Simulates processing a request and intentionally triggers an error if required parameters are missing
     * or invalid. The exception message is deliberately verbose to expose internal sensitive data.
     */
    private void processRequest(HttpServletRequest request, HttpServletResponse response) throws Exception {
        String action = request.getParameter("action");
        if (action == null || !action.equals("processData")) {
            // Construct an exception message that includes multiple internal details.
            throw new Exception("Invalid request parameters. Expected action=processData. "
                    + "Received action=" + action + ". "
                    + "DB Connection: jdbc:mysql://prod-db-server:3306/prodDB?user=prodUser&password=ProdPass!@#; "
                    + "API Key: 12345-ABCDE-SECRET; "
                    + "Config Path: /opt/app/config/secrets.conf; "
                    + "OS: " + System.getProperty("os.name") + "; "
                    + "Java Version: " + System.getProperty("java.version") + ".");
        }
        // Normal processing logic (not reached in this demonstration).
        response.getWriter().println("Request processed successfully!");
    }
}
