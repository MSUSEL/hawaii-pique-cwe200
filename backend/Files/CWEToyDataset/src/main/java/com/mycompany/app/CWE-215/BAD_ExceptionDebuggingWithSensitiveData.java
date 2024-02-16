public class BAD_ExceptionDebuggingWithSensitiveData {
    // This example demonstrates the risky practice of including sensitive data within exception handling for debugging purposes.

    public static void processPayment(String creditCardNumber, double amount) {
        try {
            // Simulate payment processing that might fail
            if (amount <= 0) {
                throw new IllegalArgumentException("Amount must be positive.");
            }
            System.out.println("Processing payment...");
            // Further processing logic...
        } catch (Exception e) {
            // Insecure: Logging sensitive information during exception handling
            System.out.println("DEBUG: Failed to process payment with card " + creditCardNumber + ": " + e.getMessage());
        }
    }

    public static void main(String[] args) {
        processPayment("1234-5678-9012-3456", -100);
    }
}
