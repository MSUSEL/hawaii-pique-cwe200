import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.http.HttpStatus;

@RestController
public class BAD_ExposedExceptionDetailsAPI {

    @GetMapping("/api/data")
    public String getData() {
        String db_user = "CWE-200-user";
        try {
            // Simulate an operation that may fail
            simulateDatabaseOperation(db_user);
            return "Data fetched successfully";
        } catch (Exception e) {
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "Failed to fetch data: " + e.getMessage(), e);
        }
    }

    private void simulateDatabaseOperation(String db_user) throws Exception {
        // Simulated failure point
        throw new Exception("Database connection error: Credentials for " + db_user + " not valid or expired");
    }
}
