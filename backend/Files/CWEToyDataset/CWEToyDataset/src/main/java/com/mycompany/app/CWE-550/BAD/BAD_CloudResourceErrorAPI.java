import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.http.HttpStatus;

@RestController
public class BAD_CloudResourceErrorAPI {

    @GetMapping("/api/resource")
    public String getResourceDetails() {
        String username = "CWE-200";
        String resourceID = "12345";
        try {
            // Attempt to retrieve cloud resource that may fail
            return getCloudResource(username, resourceID);
        } catch (Exception e) {
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "Failed to retrieve cloud resource: " + e.getMessage(), e);
        }
    }

    private String getCloudResource(String username, String resourceID) throws Exception {
        throw new Exception("Access denied for resource ID " + resourceID +", with username - " + username + ", on AWS due to improper IAM permissions.");
    }
}
