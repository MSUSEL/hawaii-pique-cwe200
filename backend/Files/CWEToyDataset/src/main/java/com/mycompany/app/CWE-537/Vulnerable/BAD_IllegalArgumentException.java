import java.util.logging.*;

public class BAD_IllegalArgumentException {
    private static final Logger logger = Logger.getLogger("RegistrationLogger");

    public void registerUser(String username, String email, String password) {
        try {
            if (!email.contains("@")) {
                throw new IllegalArgumentException("Email format is incorrect.");
            }
            // Logic to register user
        } catch (IllegalArgumentException e) {
            logger.warning("Registration attempt failed for email: " + email + ". Error: " + e.getMessage());
            throw new RuntimeException("Invalid email format.");
        }
    }
}
