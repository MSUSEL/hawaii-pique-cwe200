public class BAD_CommandLinePasswordExposure {
    public static void main(String[] args) {
        try {
            String psswrd = "secretPassword123";
            // Running a system command that includes a password in the command line
            Runtime.getRuntime().exec(new String[]{"sh", "-c", "somecommand --password " + psswrd});
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
