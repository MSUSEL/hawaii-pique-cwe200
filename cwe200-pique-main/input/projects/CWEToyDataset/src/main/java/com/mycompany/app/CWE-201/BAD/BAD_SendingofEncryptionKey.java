import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;

public class BAD_SendingofEncryptionKey extends HttpServlet {

    protected void doGet(HttpServletRequest request, HttpServletResponse response) throws IOException {
        String encryptionKey = getEncryptionKey();

        response.setContentType("application/json");
        response.getWriter().write("{\"success\": true, \"key\": \"" + encryptionKey + "\"}");
        response.getWriter().close();
    }

    private String getEncryptionKey() {
        return "4a7d1ed414474e4033ac29ccb8653d9b!";
    }

}
