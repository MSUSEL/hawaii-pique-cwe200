import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;

public class BAD_ConfigExposureServlet extends HttpServlet {
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws IOException {
        String filePath = "/etc/myapp/config.xml";
        try {
            configureService(filePath);
        } catch (Exception e) {
            response.sendError(HttpServletResponse.SC_INTERNAL_SERVER_ERROR, "Configuration error loading " + e.getMessage());

        }
    }

    private void configureService(String filePath) throws Exception {
        throw new Exception("Error loading configuration for file: " + filePath + ". Ensure the file exists and is accessible.");
    }
}
