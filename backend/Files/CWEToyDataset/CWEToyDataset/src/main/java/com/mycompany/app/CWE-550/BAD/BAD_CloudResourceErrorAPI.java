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
        String AWSKey = "aws-key-12345";
        try {
            return getCloudResource(username, resourceID, AWSKey);
        } catch (ResponseStatusException e) {
            return "Failed to retrieve cloud resource: " + e.getMessage();

        }
    }

    private String getCloudResource(String username, String resourceID, String AWSKey) throws ResponseStatusException {
        throw new ResponseStatusException(
            HttpStatus.FORBIDDEN,
            "Access denied for resource ID " + resourceID 
            + ", with username " + username + " and AWSKey " + AWSKey
            + " on AWS due to improper IAM permissions."
        );
    }
}