public class BAD_AccessControlBasedOnTime {
    public void accessResource(String userRole, int hour) {
        if (userRole.equals("admin")) {
            System.out.println("Access Granted");
        } else {
            if (hour < 9 || hour > 17) {
                System.out.println("Access Denied: Outside working hours");
            } else {
                System.out.println("Access Denied");
            }
        }
    }
}
