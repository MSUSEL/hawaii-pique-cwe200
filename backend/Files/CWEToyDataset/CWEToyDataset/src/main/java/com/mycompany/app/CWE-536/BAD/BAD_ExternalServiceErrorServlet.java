import javax.net.ssl.HttpsURLConnection;
import javax.servlet.*;
import javax.servlet.http.*;

import jakarta.servlet.ServletException;

import java.io.*;
import java.net.URL;

public class BAD_ExternalServiceErrorServlet extends HttpServlet {

    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws IOException {
        
        try {
            URL url = new URL("https://api.example.com/data");
            String apiKey = request.getParameter("apiKey");
            if (apiKey == null || apiKey.isEmpty()) {
                throw new ServletException("Invalid API key provided. " + apiKey);
            }
            HttpsURLConnection connection = (HttpsURLConnection) url.openConnection();
            // Assume connection setup and request...

            int responseCode = connection.getResponseCode();
            if (responseCode != HttpsURLConnection.HTTP_OK) {
                throw new ServletException("Failed to fetch data from API using API key: " + apiKey);
            }
        } catch (ServletException e) {
            System.err.println(e.getMessage());
        }
    }

}
