public class JavaRuntimeErrorMessageContainingSensitiveInformation {

    // Example 1 - Insecure code
    public static void insecureRuntimeError(String[] args) {
        try {
            int result = 1 / 0; // Simulating a runtime error
            System.out.println("Result: " + result);
        } catch (Exception e) {
            // Exposing sensitive information in the error message
            System.out.println("Error: " + e.getStackTrace());
        }
    }

    // Example 2 - Secure code
    public static void secureRuntimeError(String[] args) {
        try {
            int result = 1 / 0; // Simulating a runtime error
            System.out.println("Result: " + result);
        } catch (Exception e) {
            // Just Printing a generic error message
            System.out.println("Error");
        }
    }
}
