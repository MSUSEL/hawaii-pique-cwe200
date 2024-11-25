import java.io.IOException;
import javax.servlet.http.HttpServletResponse;

public class BAD_ExposeSessionIDInHeaders {
    public void exposeSessionID(HttpServletResponse response, String sessionID) {
        response.setHeader("X-Session-ID", sessionID);
        response.setContentType("text/html");
        try {
            response.getWriter().println("Session details " + sessionID + " are set in headers.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}


