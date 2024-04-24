public class GOOD_SecureLoggingWithCodes {
    // Mitigation: Using error codes and secure logging practices to avoid disclosing sensitive information.
    // Detailed error information is logged internally with a reference code, while a generic message is shown to the user.

    public static String processPayment(String accountNumber, double amount) {
        try {
            // Placeholder for payment processing logic that might fail
            validatePayment(accountNumber, amount);
            // Simulate successful payment processing
            return "Payment processed successfully.";
        } catch (Exception e) {
            // Secure: Log the detailed error internally with an error code
            String errorCode = logErrorInternally(e);
            // Provide the user with a generic message along with a reference code
            return "An error occurred processing your payment. Please contact support with error code: " + errorCode;
        }
    }

    private static void validatePayment(String accountNumber, double amount) throws Exception {
        // Simulated validation logic that can throw an exception
        if (amount <= 0) {
            throw new IllegalArgumentException("Amount must be greater than 0.");
        }
        // Additional validation checks...
    }

    private static String logErrorInternally(Exception e) {
        // Simulated function to log the error internally and return a unique error code
        String uniqueErrorCode = "ERR" + System.currentTimeMillis(); // Simplified example of generating an error code
        System.err.println("Error logged with code " + uniqueErrorCode + ": " + e.getMessage());
        return uniqueErrorCode;
    }

    public static void main(String[] args) {
        // Example usage
        System.out.println(processPayment("123456789", -100.00));
    }
}
