public class BAD_TokenExposureViaEnvironment {
    public static void main(String[] args) {
        try {
            String envToken = "WhisperingPines456";

            ProcessBuilder builder = new ProcessBuilder("myCommand");
            builder.environment().put("SECRET_TOKEN", envToken);
            Process process = builder.start();
            process.waitFor();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
