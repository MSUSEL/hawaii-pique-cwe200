public class GOOD_ProcessMonitoring {
    public void monitorProcess(String processName) {
        try {
            // Monitor process logic
            throw new Exception("Process interruption detected.");
        } catch (Exception e) {
            System.err.println("Monitoring error occurred.");
            throw new RuntimeException("Critical process monitoring failure. Immediate action required.");
        }
    }

    public static void main(String[] args) {
        new GOOD_ProcessMonitoring().monitorProcess("DatabaseSyncJob");
    }
}