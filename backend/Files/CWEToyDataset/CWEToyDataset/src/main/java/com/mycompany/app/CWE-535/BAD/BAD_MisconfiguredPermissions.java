public class BAD_MisconfiguredPermissions {

    public static void main(String[] args) {
        try {
            String scriptPath = "/path/to/sensitive/script.sh";
            ProcessBuilder pb = new ProcessBuilder("bash", scriptPath);
            Process process = pb.start();
            process.waitFor();

            System.out.println("Error Output: " + new String(process.getErrorStream().readAllBytes()));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
