public class BAD_NetworkConfigCredentialsExposure {
    public static void main(String[] args) {
        try {
            String vpnPassword = args[0];
            // Command to configure VPN with a password exposed directly in the command line
            Runtime.getRuntime().exec(new String[]{"vpnsetup", "--username", "admin", "--password", vpnPassword});
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
