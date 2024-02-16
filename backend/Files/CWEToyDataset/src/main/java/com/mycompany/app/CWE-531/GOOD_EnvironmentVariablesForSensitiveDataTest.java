public class GOOD_EnvironmentVariablesForSensitiveDataTest {

    public static void main(String[] args) {
        // Fetching sensitive data from environment variables instead of hardcoding
        String username = System.getenv("TEST_USERNAME");
        String password = System.getenv("TEST_PASSWORD");

        if(!login(username, password)) {
            System.out.println("Login failed");
        } else {
            System.out.println("Login successful");
        }
    }

    private static boolean login(String username, String password) {
        // Placeholder for login logic, assuming successful login for demonstration
        return true;
    }
}
