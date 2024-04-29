import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.assertEquals;
import java.util.Properties;


public class GOOD_ApiEndpointSecurityTest {
    private Properties properties;
    private ApiClient client;

    @BeforeEach
    public void setUp() {
        this.client = new ApiClient();
        this.properties = new Properties();
        loadTestProperties();
        client.setAuthToken(properties.getProperty("apiAuthToken"));
    }

    private void loadTestProperties() {
        properties.setProperty("apiAuthToken", "test-token");
    }

    @Test
    public void testSecureEndpoint() {
        Response response = client.makeSecureRequest();
        assertEquals(response.getStatusCode(), 200); // Assert successful response
    }

    class ApiClient {
        private String authToken;
    
        public void setAuthToken(String token) {
            this.authToken = token;
        }
    
        public Response makeSecureRequest() {
            return new Response(this.authToken.equals(System.getenv("API_AUTH_TOKEN")) ? 200 : 401);
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
