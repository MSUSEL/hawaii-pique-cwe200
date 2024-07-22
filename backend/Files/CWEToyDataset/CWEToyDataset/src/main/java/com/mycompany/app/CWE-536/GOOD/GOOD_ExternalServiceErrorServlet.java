import javax.net.ssl.HttpsURLConnection;
import javax.servlet.*;
import javax.servlet.http.*;
import java.io.*;
import java.net.URL;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class GOOD_ExternalServiceErrorServlet extends HttpServlet {
    // Setup Log4j Logger
    private static final Logger LOGGER = LogManager.getLogger(GOOD_ExternalServiceErrorServlet.class);

    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        try {
            URL url = new URL("https://api.example.com/data");
            HttpsURLConnection connection = (HttpsURLConnection) url.openConnection();
            // Assume connection setup and request...

            int responseCode = connection.getResponseCode();
            if (responseCode != HttpsURLConnection.HTTP_OK) {
                throw new ServletException("Failed to fetch data from API: HTTP " + responseCode);
            }
        } catch (IOException e) {
            throw new ServletException("Service error while fetching data.");
        }
    }

    // Function to call doGet and handle ServletException, logging details
    public void handleRequest(HttpServletRequest request, HttpServletResponse response) {
        try {
            doGet(request, response);
        } catch (ServletException e) {
            try {
                LOGGER.warn("Caught ServletException.");
                response.getWriter().println("An error occurred. Please try again later.");
            } catch (IOException ioException) {
                LOGGER.error("Failed to write to response.");
            }
        } catch (IOException e) {
            // Log general I/O errors
            LOGGER.warn("General I/O error.");
        }
    }
}
