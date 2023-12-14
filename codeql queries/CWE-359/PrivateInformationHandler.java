import java.io.IOException;
import java.util.logging.Logger;
import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

@WebServlet("/PrivateInformationHandler")
public class PrivateInformationHandler extends HttpServlet {
    private static final Logger logger = Logger.getLogger(PrivateInformationHandler.class.getName());

    protected void doGet(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {
        String address = request.getParameter("Address1");
        logger.info("User has address: " + address);
    }
}