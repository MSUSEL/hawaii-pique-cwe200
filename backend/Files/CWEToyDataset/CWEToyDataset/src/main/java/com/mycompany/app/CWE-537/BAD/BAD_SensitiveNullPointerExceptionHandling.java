import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.ServletException;
import java.io.IOException;
import java.io.PrintWriter;

public class BAD_SensitiveNullPointerExceptionHandling extends HttpServlet {

    @Override
    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, NullPointerException {
        String accountKey = null;
        
        if (request.getParameter("secretKey") != null) {
            accountKey = request.getParameter("secretKey");
        }
        else{
            throw new NullPointerException(accountKey + " is null.");
        }
    }
    
    private static String getSecretKey() {
        // Simulate a failure in secret key retrieval
        return null;
    }
}
