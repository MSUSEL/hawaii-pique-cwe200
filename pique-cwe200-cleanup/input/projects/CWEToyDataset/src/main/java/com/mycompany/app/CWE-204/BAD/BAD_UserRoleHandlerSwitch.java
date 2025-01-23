public class BAD_UserRoleHandlerSwitch {
    public enum UserRole {
        ADMIN, USER, GUEST
    }

    public String handleRequest(UserRole role, String action) {
        String response;
        switch (role) {
            case ADMIN:
                response = performAdminAction(action);
                break;
            case USER:
                response = performUserAction(action);
                break;
            case GUEST:
            default:
                response = "Access denied";
                break;
        }
        return response;
    }

    private String performAdminAction(String action) {
        // Admin-specific operations
        return "Admin action performed: " + action;
    }

    private String performUserAction(String action) {
        // User-specific operations
        return "User action performed: " + action;
    }
}