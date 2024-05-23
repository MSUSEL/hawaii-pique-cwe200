public class GOOD_EncapsulatedSecurityContext {
    // Illustrates securely executing external processes or scripts within a controlled security context,
    // mitigating the risk of exposing sensitive information through error messages.

    public static void main(String[] args) {
        try {
            String[] command = {"sh", "-c", "secureScript.sh"};
            ProcessBuilder pb = new ProcessBuilder(command);
            Process process = pb.start();

            // Securely capturing error stream output
            String errorOutput = new String(process.getErrorStream().readAllBytes());

            // Analyze and sanitize any potential sensitive information from the error output
            String sanitizedError = sanitizeErrorOutput(errorOutput);

            // Only display or log sanitized error information
            System.out.println("Sanitized Error: " + sanitizedError);
        } catch (Exception e) {
            System.out.println("An error occurred, please try again.");
        }
    }

    private static String sanitizeErrorOutput(String errorOutput) {
        // Implement sanitization logic to remove or mask sensitive information
        // This might involve regex checks for known sensitive patterns, keywords, or data formats.
        return errorOutput.replaceAll("sensitivePattern", "[REDACTED]");
    }
}
