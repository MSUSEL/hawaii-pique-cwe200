import javax.servlet.http.*;
import javax.servlet.*;
import java.io.*;

// BAD: An application component that inadvertently allows directory listing of uploaded documents.
public class BAD_DocumentDirectoryListingServlet extends HttpServlet {
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        String documentPath = "/uploaded-documents";
        File directory = new File(getServletContext().getRealPath(documentPath));
        PrintWriter out = response.getWriter();

        response.setContentType("text/html");
        out.println("<h2>Document Directory Listing:</h2>");
        out.println("<ul>");

        // Listing all files and directories within the uploaded documents directory
        for (File file : directory.listFiles()) {
            // Generating a clickable link for each file/directory (insecure exposure of directory contents)
            out.println("<li><a href='" + documentPath + "/" + file.getName() + "'>" + file.getName() + "</a></li>");
        }

        out.println("</ul>");
    }
}
