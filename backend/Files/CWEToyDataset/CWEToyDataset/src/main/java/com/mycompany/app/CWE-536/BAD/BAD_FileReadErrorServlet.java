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
            response.getWriter().write("Error reading file: " + file.getAbsolutePath());
        }
    }
}
