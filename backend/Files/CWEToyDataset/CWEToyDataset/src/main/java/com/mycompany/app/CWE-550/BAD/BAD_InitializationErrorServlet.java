import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;

public class BAD_InitializationErrorServlet extends HttpServlet {
    @Override
    public void init() throws ServletException {
        try {
            loadConfiguration();
        } catch (Exception e) {
            // BAD practice: Exposing internal configuration details during initialization failure
            System.out.println("Initialization failed due to configuration error in " + getServletContext().getRealPath("/WEB-INF/config.properties") + ": " + e.getMessage());
        }
    }

    private void loadConfiguration() throws Exception {
        // Configuration loading logic
        throw new Exception("Configuration file missing.");
    }
}
