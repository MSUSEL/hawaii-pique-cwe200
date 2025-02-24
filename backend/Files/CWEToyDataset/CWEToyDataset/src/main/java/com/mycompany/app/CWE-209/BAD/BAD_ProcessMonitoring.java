public class BAD_ProcessMonitoring {
    public void monitorProcess(String processName) {
        try {
            // Simulated process monitoring logic with additional sensitive internal details.
            String processID = "ProcID-98765";
            String machineIP = "192.168.1.101";
            String osVersion = System.getProperty("os.version");
            String osName = System.getProperty("os.name");
            String javaRuntime = System.getProperty("java.runtime.version");
            String logFilePath = "/var/log/process_monitoring.log";
            String envDetails = System.getenv("SECRET_ENV_VAR"); // Example of a sensitive environment variable

            // The exception now includes several internal configuration details.
            throw new Exception("Critical process failure detected: " + processName +
                                ". Process ID: " + processID +
                                ", Machine IP: " + machineIP +
                                ", OS: " + osName + " " + osVersion +
                                ", Java Runtime: " + javaRuntime +
                                ", Log File: " + logFilePath +
                                ", Env Var SECRET_ENV_VAR: " + envDetails);
        } catch (Exception e) {
            // Log detailed error message exposing internal data.
            System.err.println("Monitoring error: " + e.getMessage());
            throw new RuntimeException("Critical monitoring failure. Escalate immediately for process: " + processName +
                                       ". Internal details exposed in error message.", e);
        }
    }

    public static void main(String[] args) {
        new BAD_ProcessMonitoring().monitorProcess("DatabaseSyncJob");
    }
}
