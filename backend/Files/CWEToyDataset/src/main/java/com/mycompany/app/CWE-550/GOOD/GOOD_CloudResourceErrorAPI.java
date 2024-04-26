import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.http.HttpStatus;

@RestController
public class GOOD_CloudResourceErrorAPI {

    @GetMapping("/api/resource")
    public String getResourceDetails() {
        String username = "CWE-200-test";
        String resourseID = "12345";
        try {
            // Attempt to retrieve cloud resource that may fail
            return getCloudResource(username, resourseID);
        } catch (Exception e) {
            // Insecure: Exposing cloud resource access error to the client
            throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, "Failed to retrieve cloud resource: " + e.getMessage());
        }
    }

    private String getCloudResource(String username, String resourceID) throws Exception {
        throw new Exception("Access denied for resource on AWS due to improper IAM permissions.");
    }
}
