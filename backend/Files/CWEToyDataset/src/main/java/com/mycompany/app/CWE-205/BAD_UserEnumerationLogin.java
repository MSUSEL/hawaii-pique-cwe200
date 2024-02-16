public class BAD_UserEnumerationLogin {
    // Vulnerability: Observable discrepancy in login function allows user enumeration.
    // Different messages for "user does not exist" and "incorrect password" can help attackers identify valid users.

    public static String login(String username, String password) {
        if (!userExists(username)) {
            return "User does not exist."; // Reveals that the user is not in the system
        } else if (!passwordCorrect(username, password)) {
            return "Incorrect password for user."; // Implies the user exists but the password is wrong
        }
        return "Login successful.";
    }

    private static boolean userExists(String username) {
        // Placeholder for user existence check
        return "admin".equals(username);
    }

    private static boolean passwordCorrect(String username, String password) {
        // Placeholder for password correctness check
        return "admin".equals(username) && "password".equals(password);
    }

    public static void main(String[] args) {
        System.out.println(login("admin", "wrongpassword")); // Incorrect password for user.
        System.out.println(login("nonexistent", "password")); // User does not exist.
    }
}
