public class BAD_DockerRunSensitiveInfoExposure {
    public static void main(String[] args) {
        try {
            String configSecret = "config123";
            String dbPassword = "SuperSecretDBPass!";
            String apiKey = "APIKEY-1234567890";
            String encryptionKey = "EncryptKey-Alpha-2025";
            String internalCert = "/etc/ssl/certs/internal_cert.pem";

            String[] command = {
                "docker", "run",
                "-e", "CONFIG_SECRET=" + configSecret,
                "-e", "DB_PASSWORD=" + dbPassword,
                "-e", "API_KEY=" + apiKey,
                "-e", "ENCRYPTION_KEY=" + encryptionKey,
                "-e", "INTERNAL_CERT=" + internalCert,
                "myapp:latest"
            };

            // Execute the Docker command. On many operating systems,
            // the command line and environment variables are visible to other users.
            Process process = Runtime.getRuntime().exec(command);
            process.waitFor();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
