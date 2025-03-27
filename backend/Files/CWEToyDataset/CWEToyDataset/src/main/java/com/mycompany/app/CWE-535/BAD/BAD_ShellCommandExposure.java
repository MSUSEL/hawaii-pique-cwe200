public class BAD_ShellCommandExposure {

    public static void main(String[] args) {
        try {
            String filePath = "/var/backups/financeapp/data/2024-07-25/backup.tar.gz;";
            String[] command = {"sh", "-c", "cp ", filePath};
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
