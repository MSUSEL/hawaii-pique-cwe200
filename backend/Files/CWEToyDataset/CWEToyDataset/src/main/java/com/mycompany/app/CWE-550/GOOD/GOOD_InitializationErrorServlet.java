import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;

public class GOOD_InitializationErrorServlet extends HttpServlet {
    @Override
    public void init() throws ServletException {
        try {
            loadConfiguration();
        } catch (Exception e) {
            // BAD practice: Exposing internal configuration details during initialization failure
            throw new ServletException("Initialization failed due to configuration error.");
        }
    }

    private void loadConfiguration() throws Exception {
        // Configuration loading logic
        throw new Exception("Configuration file missing.");
    }
}
