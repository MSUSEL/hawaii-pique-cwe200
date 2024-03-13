public class GOOD_EnhancedDebugWithoutSensitiveData {
    // Demonstrates a creative and secure method of debugging without exposing sensitive information.

    private static final boolean DEBUG_ENABLED = System.getenv("DEBUG_MODE").equals("true");

    public static void main(String[] args) {
        String userId = System.getProperty("username", "defaultUser");
        performAction(userId);
    }

    private static void performAction(String userId) {
        try {
            // Simulated action that could fail
            throw new RuntimeException("Action failed due to system error.");
        } catch (RuntimeException e) {
            logDebugInfo("Action failure for userId: " + userId, e);
        }
    }

    private static void logDebugInfo(String message, Exception e) {
        if (DEBUG_ENABLED) {
            // Secure: Log generic message and sanitized exception details without sensitive data
            System.out.println(message);
            System.out.println("Error: " + sanitizeExceptionMessage(e.getMessage()));
            // Optionally, detailed exception stack trace can be logged to a secure, restricted log file
        }
    }

    private static String sanitizeExceptionMessage(String errorMessage) {
        // Implement sanitization logic to remove or mask sensitive information from the error message
        return errorMessage.replaceAll("sensitivePattern", "[REDACTED]");
    }
}
