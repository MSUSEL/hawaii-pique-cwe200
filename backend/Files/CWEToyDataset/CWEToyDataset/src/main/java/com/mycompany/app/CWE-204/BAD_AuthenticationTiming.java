// This example simulates an insecure authentication method that introduces a timing discrepancy

public class BAD_AuthenticationTiming {

    public static void main(String[] args) {
        // Fetching sensitive details from environment variables for demonstration
        String validUsername = System.getenv("VALID_USER");

        // Example of an authentication attempt with a wrong password
        authenticate(validUsername, "wrongPassword");
    }

    public static boolean authenticate(String attemptUsername, String attemptPassword) {
        long startTime = System.currentTimeMillis();

        // Check if the username matches the expected value
        if (attemptUsername.equals(System.getenv("VALID_USER"))) {
            // Simulate processing time for a valid username
            try {
                Thread.sleep(2000); // Delay for valid username
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }

            if (attemptPassword.equals(System.getenv("USER_PASSWORD"))) {
                System.out.println("Authenticated successfully.");
                return true;
            }
        }
        
        // Minimal processing time for invalid usernames or passwords
        long endTime = System.currentTimeMillis();
        System.out.println("Authentication failed.");

        return false;
    }
}
