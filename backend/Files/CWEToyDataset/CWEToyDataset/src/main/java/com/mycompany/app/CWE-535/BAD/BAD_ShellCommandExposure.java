public class BAD_ShellCommandExposure {

    public static void main(String[] args) {
        try {
            String[] command = {"sh", "-c", "cp /path/to/sensitive/file /backup/location"};
            Process process = Runtime.getRuntime().exec(command);
            int exitCode = process.waitFor();

            if (exitCode != 0) {
                System.out.println("Error occurred: " + new String(process.getErrorStream().readAllBytes()));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
