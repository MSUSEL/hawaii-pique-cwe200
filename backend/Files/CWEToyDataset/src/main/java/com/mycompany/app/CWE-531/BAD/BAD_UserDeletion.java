import org.junit.Test;
import static org.junit.Assert.assertTrue;

class UserDeletion {
    public boolean deleteUser(String userId, String adminAuthToken) {
        // Assume adminAuthToken is required to authenticate deletion operation
        return adminAuthToken.equals("adminTokenXYZ");
    }
}

public class BAD_UserDeletion {
    @Test
    public void testDeleteUser() {
        String userId = "user123";
        String adminAuthToken = "adminTokenXYZ"; // Sensitive admin token exposed
        UserDeletion deletion = new UserDeletion();
        assertTrue(deletion.deleteUser(userId, adminAuthToken));
    }
}
