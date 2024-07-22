public class GOOD_AccessControlError {
    public void validateUserAccess(String userName, int accessLevel) {
        try {
            if (accessLevel < 1 || accessLevel > 5) {
                throw new IllegalArgumentException("Access level is out of valid range.");
            }
            // Access validation logic
        } catch (IllegalArgumentException e) {
            System.err.println("Access level validation failed.");
            throw new SecurityException("Invalid access attempt detected.");
        }
    }

    public static void main(String[] args) {
        new GOOD_AccessControlError().validateUserAccess("adminUser", 0);
    }
}
