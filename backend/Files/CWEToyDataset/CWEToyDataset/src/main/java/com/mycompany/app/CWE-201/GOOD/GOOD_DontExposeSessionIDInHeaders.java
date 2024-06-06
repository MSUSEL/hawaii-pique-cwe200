import java.io.IOException;
import javax.servlet.http.Cookie;
import javax.servlet.http.HttpServletResponse;

public class GOOD_DontExposeSessionIDInHeaders {
    public void dontExposeSessionID(HttpServletResponse response, String sessionID) {
        // GOOD: Store the session ID in a secure, HttpOnly cookie
        Cookie sessionCookie = new Cookie("sessionID", sessionID);
        sessionCookie.setSecure(true);
        sessionCookie.setHttpOnly(true);
        response.addCookie(sessionCookie);
        response.setContentType("text/html");
        try {
            response.getWriter().println("Session details are set in a secure cookie.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}