public class BAD_NetworkConfigCredentialsExposure {
    public static void main(String[] args) {
        try {
            String vpnPassword = args[0];
            // Command to configure VPN
            Runtime.getRuntime().exec(new String[]{"vpnsetup", "--username", "admin", "--password", vpnPassword});
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
