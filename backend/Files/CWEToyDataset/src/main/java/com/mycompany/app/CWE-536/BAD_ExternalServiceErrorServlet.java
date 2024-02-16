import javax.net.ssl.HttpsURLConnection;
import javax.servlet.*;
import javax.servlet.http.*;
import java.io.*;
import java.net.URL;

public class BAD_ExternalServiceErrorServlet extends HttpServlet {
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        try {
            URL url = new URL("https://api.example.com/data");
            HttpsURLConnection connection = (HttpsURLConnection) url.openConnection();
            // Assume connection setup and request...

            int responseCode = connection.getResponseCode();
            if (responseCode != HttpsURLConnection.HTTP_OK) {
                throw new IOException("Failed to fetch data: HTTP " + responseCode);
            }
        } catch (IOException e) {
            response.getWriter().write("Service error: " + e.getMessage());
        }
    }
}
