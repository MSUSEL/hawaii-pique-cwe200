public class BAD_SecurityHandler {
    public void handleSecurityEvent(String eventType, String decryptionKey) {
        SignalCatcher.logSignal("Security event: " + eventType + " using Key: " + decryptionKey);
    }
}
