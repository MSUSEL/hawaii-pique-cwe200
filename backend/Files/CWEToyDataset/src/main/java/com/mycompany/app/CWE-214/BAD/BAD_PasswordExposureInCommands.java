public class BAD_PasswordExposureInCommands {
    public static void main(String[] args) {
        try {
            // Exposing a password in a system command is a severe security risk
            String password = "mySecretPass123";
            Process process = Runtime.getRuntime().exec("curl -u admin:" + password + " http://example.com/data");
            process.waitFor();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}