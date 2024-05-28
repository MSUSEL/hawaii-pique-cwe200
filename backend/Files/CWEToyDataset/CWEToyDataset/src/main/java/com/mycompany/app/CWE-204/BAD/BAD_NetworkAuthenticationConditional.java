public class BAD_NetworkAuthenticationConditional {
    public void authenticate(String deviceIp) {
        if (deviceIp.equals("192.168.1.1")) {
            System.out.println("Authentication Successful: Device recognized");
        } else {
            if (deviceIp.startsWith("192.168")) {
                System.out.println("Authentication Failed: Device not recognized in local network");
            } else {
                System.out.println("Authentication Failed: Unknown device");
            }
        }
    }
}
