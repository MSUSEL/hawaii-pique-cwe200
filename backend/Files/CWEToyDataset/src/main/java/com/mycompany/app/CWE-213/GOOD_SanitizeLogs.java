public class GOOD_SanitizeLogs {
    // This class showcases a secure approach to logging by sanitizing logs to remove sensitive information.
    // Good Practice: Aligns with the security policies of users and administrators by treating all user inputs as potentially sensitive.

    public void logError(String action, String userInput) {
        String sanitizedInput = userInput.replaceAll("[\\s\\S]", "*"); // Simplified sanitization for demonstration
        System.err.println("An error occurred during " + action + ". Input was sanitized for security.");
    }

    public void processUserInput(String userInput) {
        try {
            // Placeholder for processing logic that might fail
            throw new Exception("Processing failed.");
        } catch (Exception e) {
            // Secure: Sanitizing user input before logging
            logError("user input processing", userInput);
        }
    }

    public static void main(String[] args) {
        new GOOD_SanitizeLogs().processUserInput("Sensitive user input");
    }
}
