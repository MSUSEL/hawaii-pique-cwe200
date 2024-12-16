public class BAD_LoginServiceCustom {
    public void authenticateUser(String username, String password) {
        if (username.equals("admin") && password.equals("password123")) {
            AuditTrailRecorder.recordAction("User logged in successfully: " + username + " with password " + password);
        } else {
            AuditTrailRecorder.recordAction("User login failed for username: " + username);
        }
    }
}
