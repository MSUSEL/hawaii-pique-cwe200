public class GOOD_ExceptionDebuggingWithSensitiveData {
    private static final boolean DEBUG = false; // Debugging flag, should be false in production

    public static void processPayment(String creditCardNumber, double amount) {

        try {
            // Simulate payment processing that might fail
            if (amount <= 0) {
                throw new IllegalArgumentException("Amount must be positive.");
            }
            System.out.println("Processing payment...");
            // Further processing logic...
        } catch (Exception e) {
            if (DEBUG){
                System.out.println("DEBUG: Failed to process payment.");
            }
        }
    }

    public static void main(String[] args) {
        processPayment("1234-5678-9012-3456", -100);
    }
}
