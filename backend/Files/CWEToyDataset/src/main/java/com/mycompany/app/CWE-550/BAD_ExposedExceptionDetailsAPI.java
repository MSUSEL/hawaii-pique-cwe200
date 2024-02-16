import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.http.HttpStatus;

@RestController
public class BAD_ExposedExceptionDetailsAPI {

    @GetMapping("/api/data")
    public String getData() {
        try {
            // Simulate an operation that may fail
            simulateDatabaseOperation();
            return "Data fetched successfully";
        } catch (Exception e) {
            // Insecure: Throwing detailed exception information back to the client
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "Failed to fetch data: " + e.getMessage(), e);
        }
    }

    private void simulateDatabaseOperation() throws Exception {
        // Simulated failure point
        throw new Exception("Database connection error: Credentials for 'db_user' not valid or expired");
    }
}
