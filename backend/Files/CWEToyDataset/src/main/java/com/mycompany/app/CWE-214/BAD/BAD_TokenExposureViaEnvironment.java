public class BAD_TokenExposureViaEnvironment {
    public static void main(String[] args) {
        try {
            String secretToken = "s3cr3tT0k3n";

            ProcessBuilder builder = new ProcessBuilder("myCommand");
            builder.environment().put("SECRET_TOKEN", secretToken);
            Process process = builder.start();
            process.waitFor();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
