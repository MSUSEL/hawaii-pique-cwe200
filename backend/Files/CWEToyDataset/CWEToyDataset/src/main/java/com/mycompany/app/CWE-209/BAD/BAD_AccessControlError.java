public class BAD_AccessControlError {
    public void validateUserAccess(String userName, int accessLevel) {
        try {
            if (accessLevel < 1 || accessLevel > 5) {
                throw new IllegalArgumentException("Access level " + accessLevel + " is out of valid range for " + userName);
            }
            // Access validation logic
        } catch (IllegalArgumentException e) {
            System.err.println(e.getMessage());
            throw new SecurityException("Invalid access attempt detected for user: " + userName);
        }
    }

    public static void main(String[] args) {
        new BAD_AccessControlError().validateUserAccess("adminUser", 0);
    }
}
