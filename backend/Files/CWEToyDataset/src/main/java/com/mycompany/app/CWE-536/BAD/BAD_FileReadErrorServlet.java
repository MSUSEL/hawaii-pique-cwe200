import javax.servlet.*;
import javax.servlet.http.*;
import java.io.*;

public class BAD_FileReadErrorServlet extends HttpServlet {
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String filename = request.getParameter("filename");
        File file = new File("/var/www/data/" + filename);

        try (FileInputStream fis = new FileInputStream(file);
             BufferedReader reader = new BufferedReader(new InputStreamReader(fis))) {
            // Process file...
        } catch (FileNotFoundException e) {
            // This line potentially exposes the full file path, which is a security risk.
            response.getWriter().write("Error reading file: " + file.getAbsolutePath());
        }
    }

    // Additional function to handle doGet and catch ServletException
    public void safeDoGet(HttpServletRequest request, HttpServletResponse response) {
        try {
            doGet(request, response);
        } catch (ServletException e) {
            try {
                // Log and expose the ServletException inappropriately, simulating a poor security practice
                response.getWriter().println("An internal error occurred: " + e.getMessage());
                // In real-world applications, you should avoid providing this level of detail to the end user.
                e.printStackTrace(response.getWriter()); // Extremely risky to print stack trace to the user
            } catch (IOException ioException) {
                ioException.printStackTrace(); // Log to server logs, still a bad practice in catch-all
            }
        } catch (IOException e) {
            // Handle other IOExceptions that might be thrown by the response.getWriter()
            try {
                response.getWriter().println("A general error occurred: " + e.getMessage());
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
    }
}
