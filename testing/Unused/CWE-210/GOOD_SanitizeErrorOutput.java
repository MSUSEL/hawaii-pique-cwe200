public class GOOD_SanitizeErrorOutput {
    // Mitigation: Sanitizing error messages to remove sensitive information before displaying or logging.

    public static void safeLogError(Exception e) {
        // Simulated sanitization logic to remove sensitive details from the error message
        String sanitizedMessage = e.getMessage().replaceAll("/path/to/sensitive/data", "[protected]");
        System.err.println("Sanitized error: " + sanitizedMessage);
    }

    public static void main(String[] args) {
        try {
            // Placeholder for an operation that might generate an error with sensitive information
            throw new Exception("Failed to process /path/to/sensitive/data due to permission issue.");
        } catch (Exception e) {
            safeLogError(e); // Secure: Error message is sanitized before logging.
        }
    }
}
