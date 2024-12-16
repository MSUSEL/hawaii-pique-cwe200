public class BAD_EnvVarExposure {

    public static void main(String[] args) {
        String API_KEY = args[1];
        try {
            ProcessBuilder processBuilder = new ProcessBuilder("someCommand");
            processBuilder.environment().put(API_KEY, "GoldenSunset89");
            processBuilder.start();
            System.out.println("Process invoked with sensitive information in environment variable.");
        } catch (Exception e) {
            e.printStackTrace();
            System.err.println("An error occurred while invoking the process with API key " + API_KEY);
        }
    }
}
