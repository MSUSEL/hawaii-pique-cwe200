import javax.servlet.http.*;
import javax.servlet.*;
import java.io.*;

public class BAD_InsecureDirectoryListingServlet extends HttpServlet {
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String path = this.getServletContext().getRealPath("/uploads");
        File directory = new File(path);
        PrintWriter out = response.getWriter();

        for (File file : directory.listFiles()) {
            out.println(file.getName());
        }
    }
}
