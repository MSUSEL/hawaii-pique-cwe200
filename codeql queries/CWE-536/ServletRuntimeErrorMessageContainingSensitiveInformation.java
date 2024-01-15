import java.io.IOException;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

/* Description -
A servlet error message indicates that there exists an unhandled exception in your web application code and may provide useful information to an attacker.
https://cwe.mitre.org/data/definitions/536.html
*/

public class ServletRuntimeErrorMessageContainingSensitiveInformation extends HttpServlet {
    protected void doGet(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        try {
            int result = 10 / 0; // This will throw an ArithmeticException
            response.getWriter().println("Result: " + result);
        } catch (Exception e) {
            // Exposing the unhandled exception's message in the servlet error message
            response.getWriter().println("Error: " + e.getMessage());
        }
    }
}
