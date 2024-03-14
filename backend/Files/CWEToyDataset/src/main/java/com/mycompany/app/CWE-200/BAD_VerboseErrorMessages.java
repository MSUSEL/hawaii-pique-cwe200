public class BAD_VerboseErrorMessages {
    // Demonstrates an insecure practice of displaying verbose error messages that include sensitive system details.

    public static void main(String[] args) {
        try {
            // Attempt to access a resource or perform an operation that might fail
            String username = System.getenv("username");
            String password = System.getenv("password");
            
            performSensitiveOperation(username, password);
        } catch (Exception e) {
            // Insecure: Revealing detailed error information that might include sensitive system details
            System.out.println("Error occurred: " + e.getMessage());
        }
    }

    private static void performSensitiveOperation(String username, String password) throws Exception {
        // Placeholder for an operation that could throw an exception
        throw new Exception("Failed to open connection to database at /var/lib/db, user: " + username + ", password: " + password);
    }
}
