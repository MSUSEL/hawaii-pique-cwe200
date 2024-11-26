import javax.net.ssl.HttpsURLConnection;
import javax.servlet.*;
import javax.servlet.http.*;
import java.io.*;
import java.net.URL;

public class BAD_ExternalServiceErrorServlet extends HttpServlet {

    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws IOException {

        // Retrieve API key from request (e.g., header or query parameter)
        String apiKey = request.getParameter("apiKey"); // Example: /data?apiKey=12345-SECRET-API-KEY
        if (apiKey == null || apiKey.isEmpty()) {
            response.sendError(HttpServletResponse.SC_BAD_REQUEST, "API key is missing.");
            return;
        }

        int responseCode;

        try {
            URL url = new URL("https://api.example.com/data");
            HttpsURLConnection connection = (HttpsURLConnection) url.openConnection();
            connection.setRequestProperty("Authorization", "Bearer " + apiKey);
            // Assume connection setup and request...

            responseCode = connection.getResponseCode();
            if (responseCode != HttpsURLConnection.HTTP_OK) {
                throw new ServletException("Failed to fetch data. API key: " + apiKey);
            }
        } catch (ServletException e) {
            System.err.println("Error: " + e.getMessage());
            // Send a generic error response to the client
            response.sendError(HttpServletResponse.SC_INTERNAL_SERVER_ERROR, "Unable to process the request. Please try again later.");
        }
    }
}
