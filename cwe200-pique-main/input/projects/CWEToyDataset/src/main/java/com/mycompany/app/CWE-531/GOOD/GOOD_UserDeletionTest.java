import org.junit.Test;
import static org.junit.Assert.assertTrue;

class UserDeletion {
    public boolean deleteUser(String userId, String adminAuthToken) {
        // Assume adminAuthToken is required to authenticate deletion operation
        String adminAuthTokenFromDB = System.getenv("ADMIN_AUTH_TOKEN");
        return adminAuthToken.equals(adminAuthTokenFromDB);
    }
}

public class GOOD_UserDeletionTest {
    @Test
    public void testDeleteUser() {
        String userId;
        String adminAuthToken;

        userId = System.getenv("USER_ID");
        adminAuthToken = System.getenv("ADMIN_AUTH_TOKEN");
        
        UserDeletion deletion = new UserDeletion();
        assertTrue(deletion.deleteUser(userId, adminAuthToken));
    }
}
