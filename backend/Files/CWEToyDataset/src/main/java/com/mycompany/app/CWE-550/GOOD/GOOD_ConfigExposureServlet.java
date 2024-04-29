import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;

public class GOOD_ConfigExposureServlet extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws IOException {
        String filePath = "/etc/myapp/config.xml";
        try {
            configureService(filePath);
        } catch (Exception e) {
            // Insecure: Sending configuration exception details back to the client
            response.sendError(HttpServletResponse.SC_INTERNAL_SERVER_ERROR, "Configuration error loading.");

        }
    }

    private void configureService(String filePath) throws Exception {
        throw new Exception("Error loading configuration for file. Ensure the file exists and is accessible.");
    }
}
