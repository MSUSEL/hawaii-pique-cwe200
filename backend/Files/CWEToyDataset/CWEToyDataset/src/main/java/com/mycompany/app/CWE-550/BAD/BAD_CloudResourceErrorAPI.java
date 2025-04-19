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
            return getCloudResource(username, resourceID);
        } catch (ResponseStatusException e) {
            return "Failed to retrieve cloud resource: " 
                + e.getReason() 
                + " | Debug Info: username=" + username 
                + ", resourceID=" + resourceID;
        }
    }

    private String getCloudResource(String username, String resourceID) throws ResponseStatusException {
        throw new ResponseStatusException(
            HttpStatus.FORBIDDEN,
            "Access denied for resource ID " + resourceID 
            + ", with username " + username 
            + " on AWS due to improper IAM permissions."
        );
    }
}