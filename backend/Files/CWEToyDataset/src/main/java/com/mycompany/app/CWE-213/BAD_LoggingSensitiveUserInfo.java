public class BAD_LoggingSensitiveUserInfo {
    // This class demonstrates an insecure logging practice by including sensitive user information (such as passwords) in log files during authentication attempts.

    public static boolean authenticateUser(String username, String password) {
        try {
            // Placeholder for authentication logic
            if ("admin".equals(username) && "adminPass".equals(password)) {
                System.out.println("User authenticated successfully.");
                return true;
            } else {
                throw new SecurityException("Authentication failed for user: " + username + " with password: " + password);
            }
        } catch (SecurityException e) {
            // Insecure: Logging both the username and the password, potentially exposing sensitive information in logs.
            System.err.println(e.getMessage());
            return false;
        }
    }

    public static void main(String[] args) {
        authenticateUser("admin", "wrongPassword");
    }
}
