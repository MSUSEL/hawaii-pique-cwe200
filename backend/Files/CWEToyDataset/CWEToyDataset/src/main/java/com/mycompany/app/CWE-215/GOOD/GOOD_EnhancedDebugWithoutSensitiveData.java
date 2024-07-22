public class GOOD_EnhancedDebugWithoutSensitiveData {

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
            System.out.println(message);
            System.out.println("Error: " + sanitizeExceptionMessage(e.getMessage()));
        }
    }

    private static String sanitizeExceptionMessage(String errorMessage) {
        return errorMessage.replaceAll("sensitivePattern", "[REDACTED]");
    }
}
