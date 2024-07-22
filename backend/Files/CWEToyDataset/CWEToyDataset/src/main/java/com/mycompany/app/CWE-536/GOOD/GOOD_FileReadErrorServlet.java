import javax.servlet.*;
import javax.servlet.http.*;
import java.io.*;

public class GOOD_FileReadErrorServlet extends HttpServlet {
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String filename = request.getParameter("filename");
        File file = new File("/var/www/data/" + filename);

        try (FileInputStream fis = new FileInputStream(file);
             BufferedReader reader = new BufferedReader(new InputStreamReader(fis))) {
            // Process file...
        } catch (FileNotFoundException e) {
            response.getWriter().write("Error reading file.");
        }
    }

    // Additional function to handle doGet and catch ServletException
    public void safeDoGet(HttpServletRequest request, HttpServletResponse response) {
        try {
            doGet(request, response);
        } catch (ServletException e) {
            try {
                response.getWriter().println("An internal error occurred.");
            } catch (IOException ioException) {
            }
        } catch (IOException e) {
            // Handle other IOExceptions that might be thrown by the response.getWriter()
            try {
                response.getWriter().println("A general error occurred.");
            } catch (IOException ex) {
            }
        }
    }
}
