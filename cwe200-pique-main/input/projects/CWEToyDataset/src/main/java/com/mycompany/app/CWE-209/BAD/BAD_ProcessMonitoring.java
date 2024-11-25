public class BAD_ProcessMonitoring {
    public void monitorProcess(String processName) {
        try {
            // Monitor process logic
            throw new Exception("Critical process failure detected: " + processName);
        } catch (Exception e) {
            System.err.println("Monitoring error: " + e.getMessage());
            throw new RuntimeException("Critical monitoring failure. Escalate immediately for process: " + processName, e);
        }
    }

    public static void main(String[] args) {
        new BAD_ProcessMonitoring().monitorProcess("DatabaseSyncJob");
    }
}
