import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class ExposureofInformationThroughDirectoryListing extends HttpServlet {

    private final String BASE_DIRECTORY = "/path/to/serve";

    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        String requestedPath = request.getPathInfo();

        if (requestedPath == null) {
            requestedPath = "/";
        }

        File file = new File(BASE_DIRECTORY, requestedPath);

        // If the path is a directory, list all files (vulnerable to directory listing)
        if (file.isDirectory()) {
            File[] files = file.listFiles();
            if (files != null) {
                for (File f : files) {
                    response.getWriter().println(f.getName());
                }
            }
        } else if (file.isFile()) {
            // Serve the file content
            Path filePath = file.toPath();
            Files.copy(filePath, response.getOutputStream());
        } else {
            response.sendError(HttpServletResponse.SC_NOT_FOUND);
        }
    }
}
