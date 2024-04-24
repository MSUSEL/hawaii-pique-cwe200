public class BAD_ShellCommandExposure {
    // Demonstrates insecure execution of a shell command that might expose sensitive information in error messages.

    public static void main(String[] args) {
        try {
            String[] command = {"sh", "-c", "cp /path/to/sensitive/file /backup/location"};
            Process process = Runtime.getRuntime().exec(command);
            int exitCode = process.waitFor();

            if (exitCode != 0) {
                // Insecure: Exposing potential sensitive information through error stream.
                System.out.println("Error occurred: " + new String(process.getErrorStream().readAllBytes()));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
