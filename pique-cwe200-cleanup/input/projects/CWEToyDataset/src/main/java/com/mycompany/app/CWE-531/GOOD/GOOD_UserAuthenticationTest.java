import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

public class GOOD_UserAuthenticationTest {
    private String username;
    private String password;

    @BeforeEach
    public void setup() {
        // Fetch credentials securely from environment variables
        username = System.getenv("TEST_USERNAME");
        password = System.getenv("TEST_PASSWORD");

        // Ensure that the environment variables are correctly set
        if (username == null || password == null) {
            fail("Environment variables for credentials must be set");
        }
    }

    @Test
    public void testLogin() {
        Authentication auth = new Authentication();
        assertTrue(auth.login(username, password), "This test passes if login is successful");
    }

    class Authentication {
        public boolean login(String username, String password) {
            String validUsername;
            String validPassword;
            
            validUsername = System.getenv("VALID_USERNAME");
            validPassword = System.getenv("VALID_PASSWORD");
    
            return validUsername.equals(username) && validPassword.equals(password);
        }
    }
}
