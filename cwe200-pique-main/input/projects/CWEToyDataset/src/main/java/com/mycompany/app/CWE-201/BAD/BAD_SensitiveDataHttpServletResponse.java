import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;

public class BAD_SensitiveDataHttpServletResponse extends HttpServlet {
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws IOException {
        String ssn = getSensitiveData();

        response.setContentType("text/plain");
        response.getWriter().write("Sensitive Data: " + ssn);
    }

    // Simulates retrieval of sensitive data, e.g., from a database or a file
    private String getSensitiveData() {
        // Example sensitive data, like a social security number
        return "SSN: 123-45-6789";
    }
}
