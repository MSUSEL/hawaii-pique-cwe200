public class GOOD_UniformErrorResponses {
    // Mitigation: Using uniform error messages to avoid revealing details about internal states or logic.
    // This approach prevents attackers from distinguishing between different types of errors or system behaviors.

    public static String authenticate(String username, String password) {
        // Regardless of the authentication outcome, return a generic error message.
        if (!"admin".equals(username) || !"password".equals(password)) {
            return "Authentication failed. Please try again."; // Uniform message hides the specific reason.
        }
        return "Login successful.";
    }

    public static void main(String[] args) {
        System.out.println(authenticate("admin", "wrongpassword")); // Authentication failed. Please try again.
        System.out.println(authenticate("nonexistentuser", "password")); // Same error message for non-existent user.
    }
}
