import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class BAD_ApiEndpointSecurityTest {
    @Test
    public void testSecureEndpoint() {
        ApiClient client = new ApiClient();
        String validationToken = "XYZ-780320-BHDM";
        client.setAuthToken(validationToken);
        Response response = client.makeSecureRequest();
        assertEquals(response.getStatusCode(), 200); // Assert successful response
    }


    class ApiClient {
        private String authToken;
    
        public void setAuthToken(String token) {
            this.authToken = token;
        }
    
        public Response makeSecureRequest() {
            return new Response(this.authToken.equals("EveningGlow2025") ? 200 : 401);
        }
    }
    
    class Response {
        private int statusCode;
    
        public Response(int statusCode) {
            this.statusCode = statusCode;
        }
    
        public int getStatusCode() {
            return statusCode;
        }
    }
}
