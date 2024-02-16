public class GOOD_GenericErrorMessages {
    // Illustrates a secure practice of using generic error messages to avoid exposing sensitive system details.

    public static void main(String[] args) {
        try {
            // Attempt to access a resource or perform an operation that might fail
            performSensitiveOperation();
        } catch (Exception e) {
            // Secure: Using a generic error message that does not reveal sensitive information
            System.out.println("An error occurred. Please contact support.");
        }
    }

    private static void performSensitiveOperation() {
        // Placeholder for an operation that could fail
    }
}
