import java.io.BufferedWriter;
import java.io.OutputStreamWriter;

public class GOOD_SecureSubprocessCommunication {
    // Illustrates a secure method of communicating sensitive information to subprocesses without exposing it through command-line arguments or environment variables.

    public static void main(String[] args) {
        try {
            ProcessBuilder processBuilder = new ProcessBuilder("sh");
            Process process = processBuilder.start();
            
            // Securely passing sensitive information to the subprocess through its standard input stream.
            try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(process.getOutputStream()))) {
                writer.write("secureCommand --option\n");
                writer.flush();
                // Sensitive data is passed securely; not visible in process listings.
            }
            System.out.println("Securely communicated with subprocess without exposing sensitive information.");
        } catch (Exception e) {
            System.err.println("An error has occurred.");
        }
    }
}
