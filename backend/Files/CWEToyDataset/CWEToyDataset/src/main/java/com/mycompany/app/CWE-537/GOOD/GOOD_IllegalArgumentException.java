import java.util.logging.*;

public class GOOD_IllegalArgumentException {
    private static final Logger logger = Logger.getLogger("RegistrationLogger");

    public void registerUser(String username, String email, String password) {
        try {
            if (!email.contains("@")) {
                throw new IllegalArgumentException("Email format is incorrect.");
            }
            // Logic to register user
        } catch (IllegalArgumentException e) {
            logger.warning("Registration attempt failed.");
            throw new RuntimeException("Invalid email format.");
        }
    }
}
