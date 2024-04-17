import javax.net.ssl.HttpsURLConnection;
import javax.servlet.*;
import javax.servlet.http.*;
import java.io.*;
import java.net.URL;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class BAD_ExternalServiceErrorServlet extends HttpServlet {
    // Setup Log4j Logger
    private static final Logger LOGGER = LogManager.getLogger(BAD_ExternalServiceErrorServlet.class);

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
            throw new ServletException("Service error while fetching data", e);
        }
    }

    // Function to call doGet and handle ServletException, logging details
    public void handleRequest(HttpServletRequest request, HttpServletResponse response) {
        try {
            doGet(request, response);
        } catch (ServletException e) {
            try {
                // Logging the exception details inappropriately
                LOGGER.error("Caught ServletException: " + e.getMessage(), e);
                // In real-world scenarios, ensure that sensitive information is not logged like this
                response.getWriter().println("An error occurred. Please try again later.");
            } catch (IOException ioException) {
                LOGGER.error("Failed to write to response", ioException);
            }
        } catch (IOException e) {
            // Log general I/O errors
            LOGGER.warn("General I/O error: " + e.getMessage(), e);
        }
    }
}
