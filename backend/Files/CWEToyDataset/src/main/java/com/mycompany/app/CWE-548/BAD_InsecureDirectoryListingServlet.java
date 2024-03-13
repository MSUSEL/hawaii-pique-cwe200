import javax.servlet.http.*;
import javax.servlet.*;
import java.io.*;

// BAD: Servlet that allows directory listing due to improper handling
public class BAD_InsecureDirectoryListingServlet extends HttpServlet {
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String path = this.getServletContext().getRealPath("/uploads");
        File directory = new File(path);
        PrintWriter out = response.getWriter();

        // Insecurely listing files in the directory
        for (File file : directory.listFiles()) {
            out.println(file.getName());
        }
    }
}
