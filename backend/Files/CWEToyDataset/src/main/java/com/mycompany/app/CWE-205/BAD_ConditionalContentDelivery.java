public class BAD_ConditionalContentDelivery {
    // Vulnerability: The system changes its response behavior based on the user's role,
    // allowing an observer to infer the existence or privilege level of a user based on the content delivered or the action taken.

    public static String getContentBasedOnUserRole(String userRole) {
        // Simulate fetching content for different roles with observable behavioral discrepancy
        switch (userRole) {
            case "admin":
                // Reveals through verbose logging that a user with admin privileges is accessing the system.
                System.out.println("Delivering admin-specific content.");
                return "Admin Dashboard";
            case "user":
                System.out.println("Delivering user-specific content.");
                return "User Dashboard";
            default:
                // A different response behavior for unrecognized roles might help an attacker identify valid roles.
                System.out.println("Access denied. Your role is not recognized.");
                return "Access Denied";
        }
    }

    public static void main(String[] args) {
        System.out.println(getContentBasedOnUserRole("admin")); // Clearly differentiates behavior for an admin.
        System.out.println(getContentBasedOnUserRole("guest")); // Indicates the guest role is not recognized, revealing system behavior.
    }
}
