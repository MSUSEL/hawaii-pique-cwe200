public class BAD_SessionManager {
    public void startSession(String userId, String token) {
        TraceKeeper.enterMethod("Session started for UserID: " + userId + " with Token: " + token);
        // Session logic
        TraceKeeper.exitMethod("Session ended for UserID: " + userId);
    }
}
