import java.io.IOException;
import javax.servlet.*;
import javax.servlet.http.*;
import org.apache.logging.log4j.*;

public class GOOD_IoTDeviceFirmwareUpdateServlet extends HttpServlet {
    private static final Logger logger = LogManager.getLogger(GOOD_IoTDeviceFirmwareUpdateServlet.class);

    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String deviceId = request.getParameter("deviceId");
        String firmwareVersion = request.getParameter("firmwareVersion");
        try {
            deployFirmwareUpdate(deviceId, firmwareVersion);
        } catch (ServletException e) {
            logger.error("Failed firmware update failed.");
            response.getWriter().println("Firmware update process encountered an error. Please check device status.");
            notifyTechSupport(deviceId, firmwareVersion, e.getMessage());
        }
    }

    private void deployFirmwareUpdate(String deviceId, String firmwareVersion) throws ServletException {
        // Logic to deploy firmware which might fail and needs careful exception handling
        if (firmwareVersion.equals("0.23Beta")) {
            throw new ServletException("Firmware version has known issues with this device.");
        }
        // Assume more update logic here that could fail
    }

    private void notifyTechSupport(String deviceId, String firmwareVersion, String error) {
        // Notifying technical support with detailed error info, could log sensitive information if not properly managed
        logger.info("Tech support notified.");
    }
}
