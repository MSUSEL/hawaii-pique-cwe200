import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.http.HttpStatus;

@RestController
public class BAD_UserDataHandlingAPI {

    @PostMapping("/api/users")
    public String updateUserProfile(String userData) {
        try {
            // Process user data that may include sensitive operations
            processUserData(userData);
            return "User profile updated successfully";
        } catch (Exception e) {
            // Insecure: Exposing detailed user data handling errors
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "Failed to update user profile: " + e.getMessage(), e);
        }
    }

    private void processUserData(String userdata) throws Exception {
        throw new Exception("User data validation failed due to invalid credentials - " + userdata + "field in input data");
    }
}