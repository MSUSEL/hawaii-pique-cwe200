import javax.net.ssl.HttpsURLConnection;
import javax.servlet.*;
import javax.servlet.http.*;
import java.io.*;
import java.net.URL;

public class BAD_ExternalServiceErrorServlet extends HttpServlet {

    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws IOException {
        
        int responseCode = 0;
        try {
            URL url = new URL("https://api.example.com/data");
            HttpsURLConnection connection = (HttpsURLConnection) url.openConnection();
            // Assume connection setup and request...

            responseCode = connection.getResponseCode();
            if (responseCode != HttpsURLConnection.HTTP_OK) {
                throw new ServletException();
            }
        } catch (ServletException e) {
            System.err.println("Failed to fetch data from API: HTTP " + responseCode);
        }
    }

}
