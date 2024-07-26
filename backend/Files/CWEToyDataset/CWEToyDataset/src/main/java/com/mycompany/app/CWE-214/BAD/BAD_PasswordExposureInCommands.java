public class BAD_PasswordExposureInCommands {
    public static void main(String[] args) {
        try {
            String password = "P@ssw0rd2024!";
            Process process = Runtime.getRuntime().exec("curl -u admin:" + password + " http://healthcare.com/data");
            process.waitFor();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
