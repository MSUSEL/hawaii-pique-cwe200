import java.io.IOException;
import javax.servlet.*;
import javax.servlet.http.*;
import org.apache.logging.log4j.*;

public class BAD_IoTDeviceFirmwareUpdateServlet extends HttpServlet {
    private static final Logger logger = LogManager.getLogger(BAD_IoTDeviceFirmwareUpdateServlet.class);

    protected void doPost(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String deviceId = request.getParameter("deviceId");
        String firmwareVersion = request.getParameter("firmwareVersion");
        
        deployFirmwareUpdate(deviceId, firmwareVersion);
    }

    private void deployFirmwareUpdate(String deviceId, String firmwareVersion) throws ServletException {
        // Logic to deploy firmware which might fail and needs careful exception handling
        if (firmwareVersion.equals("0.23Beta")) { 
            throw new ServletException("Firmware version " + firmwareVersion + " has known issues with device ID: " + deviceId);
        }
        // Assume more update logic here that could fail
    }


}
