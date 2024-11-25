public class BAD_CommandLinePasswordExposure {
    public static void main(String[] args) {
        try {
            String psswrd = "BlueSkyMorning";
            // Running a system command
            Runtime.getRuntime().exec(new String[]{"sh", "-c", "somecommand --password " + psswrd});
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
