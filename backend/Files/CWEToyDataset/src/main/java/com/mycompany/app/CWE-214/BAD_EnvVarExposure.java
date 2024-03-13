public class BAD_EnvVarExposure {
    // This example shows setting an environment variable with sensitive information before invoking a process.
    // Environment variables can also be viewed by other users on the same system, leading to potential exposure of sensitive data.

    public static void main(String[] args) {
        String API_KEY = args[1];
        try {
            ProcessBuilder processBuilder = new ProcessBuilder("someCommand");
            // Setting an environment variable with sensitive information
            processBuilder.environment().put(API_KEY, "secretApiKey");
            processBuilder.start();
            System.out.println("Process invoked with sensitive information in environment variable.");
        } catch (Exception e) {
            e.printStackTrace();
            System.err.println("An error occurred while invoking the process with API key " + API_KEY);
        }
    }
}
