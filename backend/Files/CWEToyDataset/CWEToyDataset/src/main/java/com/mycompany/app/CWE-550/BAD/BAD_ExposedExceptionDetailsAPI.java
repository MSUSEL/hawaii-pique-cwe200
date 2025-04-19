import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.http.HttpStatus;
import org.springframework.web.server.ResponseStatusException;
import java.util.Arrays;

@RestController
public class BAD_ExposedExceptionDetailsAPI {

    @GetMapping("/api/data")
    public String getData(@RequestParam("userId") String userId) {
        String dbUser = System.getenv("DB_USER");
        String dbPass = System.getenv("DB_PASS");
        String dbUrl = System.getenv("DB_URL");
        String query = "SELECT * FROM users WHERE id = '" + userId + "'";
        try {
            return simulateDatabaseOperation(dbUrl, dbUser, dbPass, query);
        } catch (ResponseStatusException e) {
            return "Failed to fetch data for user " + userId
                + ": " + e.getMessage()
                + " | DB URL=" + dbUrl
                + ", DB User=" + dbUser
                + ", DB Pass=" + dbPass
                + ", Query=" + query
                + " | StackTrace=" + Arrays.toString(e.getStackTrace());
        }
    }

    private String simulateDatabaseOperation(String url, String user, String pass, String query) throws ResponseStatusException {
        // Simulate a JDBC connection attempt and query execution
        throw new ResponseStatusException(HttpStatus.BAD_REQUEST,
            "Simulated SQLException: Access denied for user '" + user + "'@'" + url + "'; attempted query: " + query);
    }
}