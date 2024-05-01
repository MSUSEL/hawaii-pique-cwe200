import javax.servlet.*;
import javax.servlet.http.*;
import java.io.*;

public class BAD_KeyInfoExposureServlet extends HttpServlet {
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        File keyDir = new File("/secure/keys");
        File[] keyFiles = keyDir.listFiles();

        if (keyFiles != null) {
            PrintWriter out = response.getWriter();
            response.setContentType("text/plain");

            for (File keyFile : keyFiles) {
                // Improperly exposing encryption key file details
                out.println("Encryption Key File: " + keyFile.getName() + " Size: " + keyFile.length() + " bytes");
            }
        } else {
            response.getWriter().println("No keys found.");
        }
    }
}
