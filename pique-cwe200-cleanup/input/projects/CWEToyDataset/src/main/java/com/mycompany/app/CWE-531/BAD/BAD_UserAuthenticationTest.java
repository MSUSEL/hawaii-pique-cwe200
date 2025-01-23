import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertTrue;

class Authentication {
    public boolean login(String username, String password) {
        return "admin".equals(username) && "admin123".equals(password);
    }
}

public class BAD_UserAuthenticationTest {
    @Test
    public void testLogin() {
        String username = "admin";
        String password = "admin123"; 
        Authentication auth = new Authentication();
        assertTrue(auth.login(username, password)); // This test passes if login is successful
    }
}
