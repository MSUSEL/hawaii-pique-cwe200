import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertTrue;

class Authentication {
    public boolean login(String username, String password) {
        return "admin".equals(username) && "admin123".equals(password);
    }
}

public class BAD_UserAuthentication {
    @Test
    public void testLogin() {
        String username = "admin";
        String password = "admin123"; // Sensitive data hardcoded in test code
        Authentication auth = new Authentication();
        assertTrue(auth.login(username, password)); // This test passes if login is successful
    }
}
