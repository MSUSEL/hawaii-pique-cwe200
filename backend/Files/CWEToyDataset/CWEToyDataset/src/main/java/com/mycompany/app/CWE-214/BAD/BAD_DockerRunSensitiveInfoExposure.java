public class BAD_DockerRunSensitiveInfoExposure {
    public static void main(String[] args) {
        try {
            String configSecret = "config123Secret!";
            // Running a Docker container
            Runtime.getRuntime().exec(new String[]{"docker", "run", "-e", "CONFIG_SECRET=" + configSecret, "myapp:latest"});
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
