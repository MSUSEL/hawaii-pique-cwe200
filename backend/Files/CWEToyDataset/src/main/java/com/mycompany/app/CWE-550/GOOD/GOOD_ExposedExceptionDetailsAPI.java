import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.http.HttpStatus;

@RestController
public class GOOD_ExposedExceptionDetailsAPI {

    @GetMapping("/api/data")
    public String getData() {
        String db_user = System.getenv("DB_USER");
        try {
            // Simulate an operation that may fail
            simulateDatabaseOperation(db_user);
            return "Data fetched successfully";
        } catch (Exception e) {
            // Insecure: Throwing detailed exception information back to the client
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "Failed to fetch data: " + e.getMessage());
        }
    }

    private void simulateDatabaseOperation(String db_user) throws Exception {
        // Simulated failure point
        throw new Exception("Database connection error.");
    }
}
