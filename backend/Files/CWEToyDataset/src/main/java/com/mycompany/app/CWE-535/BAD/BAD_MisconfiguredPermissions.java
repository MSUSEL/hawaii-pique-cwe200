public class BAD_MisconfiguredPermissions {
    // Demonstrates a vulnerability where executing a script with misconfigured permissions
    // might lead to unauthorized access to sensitive information contained in error messages.

    public static void main(String[] args) {
        try {
            String scriptPath = "/path/to/sensitive/script.sh";
            ProcessBuilder pb = new ProcessBuilder("bash", scriptPath);
            Process process = pb.start();
            process.waitFor();

            // Insecure: Assuming error stream might contain sensitive information due to misconfiguration
            System.out.println("Error Output: " + new String(process.getErrorStream().readAllBytes()));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
