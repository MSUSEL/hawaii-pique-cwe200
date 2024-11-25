import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;

public class BAD_InitializationErrorServlet extends HttpServlet {
    @Override
    public void init() throws ServletException {
        String configPath = getServletContext().getRealPath("/WEB-INF/config.properties");
        try {
            loadConfiguration(configPath);
        } catch (Exception e) {
            System.out.println("Initialization failed due to configuration error in " + e.getMessage());
        }
    }

    private void loadConfiguration(String configPath) throws Exception {
        // Configuration loading logic
        throw new Exception("Configuration file missing at " + configPath);
    }
}
