import javax.servlet.*;
import javax.servlet.http.*;
import java.io.*;

public class BAD_FileReadErrorServlet extends HttpServlet {
    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws IOException, ServletException {
        String filename = request.getParameter("filename");
        File file = new File("/var/www/data/" + filename);

        try (FileInputStream fis = new FileInputStream(file);
             BufferedReader reader = new BufferedReader(new InputStreamReader(fis))) {
            // Example condition to throw ServletException
            if (file.length() == 0) {
                throw new ServletException("File is empty: " + file.getAbsolutePath());
            }
            // Process file...
        } catch (IOException e) {
            response.getWriter().write("Error reading file: " + file.getAbsolutePath());
        } catch (ServletException e) {
            response.getWriter().write(e.getMessage());
        }
    }
}
