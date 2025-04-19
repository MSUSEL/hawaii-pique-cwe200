import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.http.HttpStatus;
import org.springframework.web.server.ResponseStatusException;

@RestController
public class BAD_UserDataHandlingAPI {

    @PostMapping("/api/users")
    public String updateUserProfile(String userData) {
        try {
            processUserData(userData);
            return "User profile updated successfully";
        } catch (ResponseStatusException e) {
            return "Failed to update user profile: " + e.getMessage();
        }
    }

    private void processUserData(String userData) throws ResponseStatusException {
        throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "User data validation failed due to invalid credentials: " + userData);
    }
}