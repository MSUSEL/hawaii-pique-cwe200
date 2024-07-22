public class GOOD_EncapsulatedSecurityContext {

    public static void main(String[] args) {
        try {
            String[] command = {"sh", "-c", "secureScript.sh"};
            ProcessBuilder pb = new ProcessBuilder(command);
            Process process = pb.start();

            String errorOutput = new String(process.getErrorStream().readAllBytes());

            String sanitizedError = sanitizeErrorOutput(errorOutput);

            System.out.println("Sanitized Error: " + sanitizedError);
        } catch (Exception e) {
            System.out.println("An error occurred, please try again.");
        }
    }

    private static String sanitizeErrorOutput(String errorOutput) {
        return errorOutput.replaceAll("sensitivePattern", "[REDACTED]");
    }
}
