public class BAD_HardcodedCredentialsTest {

    public static void main(String[] args) {
        // Example of a unit test method that uses hardcoded credentials
        String username = "admin";
        String password = "admin123"; // Sensitive information hardcoded

        if(!login(username, password)) {
            System.out.println("Login failed");
        } else {
            System.out.println("Login successful");
        }
    }

    private static boolean login(String username, String password) {
        // Placeholder for login logic
        return "admin".equals(username) && "admin123".equals(password);
    }
}
