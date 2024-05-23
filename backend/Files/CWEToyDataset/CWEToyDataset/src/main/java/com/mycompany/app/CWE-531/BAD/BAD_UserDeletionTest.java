import org.junit.Test;
import static org.junit.Assert.assertTrue;

public class BAD_UserDeletionTest {
    @Test
    public void testDeleteUser() {
        String userId = "user123";
        String adminAuthToken = "adminTokenXYZ"; // Sensitive admin token exposed
        UserDeletionTest deletion = new UserDeletionTest();
        assertTrue(deletion.deleteUser(userId, adminAuthToken));
    }

    class UserDeletionTest {
        public boolean deleteUser(String userId, String adminAuthToken) {
            // Assume adminAuthToken is required to authenticate deletion operation
            return adminAuthToken.equals("adminTokenXYZ");
        }
    }
}
