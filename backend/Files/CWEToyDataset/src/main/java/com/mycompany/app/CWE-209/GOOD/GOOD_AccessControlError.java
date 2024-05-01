public class GOOD_AccessControlError {
    public void validateUserAccess(String userName, int accessLevel) {
        try {
            if (accessLevel < 1 || accessLevel > 5) {
                // Avoid including sensitive details about the user or access level in the exception message
                throw new IllegalArgumentException("Access level is out of valid range.");
            }
            // Access validation logic
        } catch (IllegalArgumentException e) {
            // Log the error internally without exposing it to the user or external logs
            System.err.println("Access level validation failed.");  // Generic error message
            // Throwing a generic SecurityException without user details
            throw new SecurityException("Invalid access attempt detected.");
        }
    }

    public static void main(String[] args) {
        new GOOD_AccessControlError().validateUserAccess("adminUser", 0);
    }
}
