public class BAD_DockerRunSensitiveInfoExposure {
    public static void main(String[] args) {
        try {
            String config = "config123";
            // Running a Docker container
            Runtime.getRuntime().exec(new String[]{"docker", "run", "-e", "CONFIG_SECRET=" + config, "myapp:latest"});
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
