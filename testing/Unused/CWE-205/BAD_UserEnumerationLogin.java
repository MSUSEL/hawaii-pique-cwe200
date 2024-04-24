public class BAD_UserEnumerationLogin {
    // Vulnerability: Observable discrepancy in login function allows user enumeration.
    // Generic messages are not used, which can help attackers identify valid users.

    public static String login(String username, String password) {
        // Fetching the "valid" username and password from environment variables instead of hardcoding
        String validUsername = System.getenv("VALID_USERNAME");
        String validPassword = System.getenv("VALID_PASSWORD");

        if (!userExists(username, validUsername)) {
            return "User does not exist."; // Reveals that the user is not in the system
        } else if (!passwordCorrect(username, password, validUsername, validPassword)) {
            return "Incorrect password for user."; // Implies the user exists but the password is wrong
        }
        return "Login successful.";
    }

    private static boolean userExists(String username, String validUsername) {
        // Checks against a fetched valid username
        return validUsername.equals(username);
    }

    private static boolean passwordCorrect(String username, String password, String validUsername, String validPassword) {
        // Checks password correctness only if the username matches the fetched valid username
        return validUsername.equals(username) && validPassword.equals(password);
    }

    public static void main(String[] args) {
        // Example usage with dynamic values
        String testUsername = "attemptUsername"; // Placeholder for user input
        String testPassword = "attemptPassword"; // Placeholder for user input

        System.out.println(login(testUsername, testPassword)); // Results vary based on environment variable values
    }
}
