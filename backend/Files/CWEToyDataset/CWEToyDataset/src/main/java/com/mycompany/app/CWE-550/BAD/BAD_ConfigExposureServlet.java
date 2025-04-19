import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.net.BindException;

public class BAD_ConfigExposureServlet extends HttpServlet {
    @Override
    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws IOException {
        String filePath = "/etc/myapp/config.xml";
        try {
            configureService(filePath);
        } catch (BindException e) {
            response.sendError(HttpServletResponse.SC_INTERNAL_SERVER_ERROR, "Configuration error: " + e.getMessage());
        }
    }

    private void configureService(String filePath) throws BindException {
        String javaHome = System.getenv("JAVA_HOME");
        String osName = System.getProperty("os.name");
        String macAddress = System.getProperty("mac_address");
        String ipAddr = System.getProperty("ip_addr");
        String sysInfo = "Environment: JAVA_HOME=" + javaHome +
                         ", OS=" + osName +
                         ", Mac Address=" + macAddress +
                         ", IP Address=" + ipAddr;
        throw new BindException("Error binding configuration for file: " + filePath +
                                ". Ensure the file exists and is accessible. " + sysInfo);
    }
}