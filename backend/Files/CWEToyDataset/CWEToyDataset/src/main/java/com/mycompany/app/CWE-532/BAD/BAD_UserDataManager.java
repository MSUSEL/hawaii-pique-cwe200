import java.util.logging.*;

public class BAD_UserDataManager {
    private static final Logger logger = Logger.getLogger("UserDataManager");

    public void storeUserData(String userId, String personalData) {
        try {
            // Simulate storing user data
            logger.fine("Storing data for user: " + userId + " with data: " + personalData); 
        } catch (Exception e) {
            logger.severe("Failed to store user data for user: " + userId + ", error: " + e.getMessage());
        }
    }
}
