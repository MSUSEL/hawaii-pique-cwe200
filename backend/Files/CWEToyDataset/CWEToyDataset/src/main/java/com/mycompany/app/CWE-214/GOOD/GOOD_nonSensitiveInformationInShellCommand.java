import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
import java.util.Base64;

public class GOOD_nonSensitiveInformationInShellCommand {
    public static void main(String[] args) {
        try {
            String command = "analyze --no-sarif --format=csv --output=results.csv";

            // Now we use the encrypted token as a command line argument
            ProcessBuilder builder = new ProcessBuilder("codeql", command);
            Process process = builder.start();
            process.waitFor();  // Wait for the process to complete
            System.out.println("Cloud deployment initiated successfully.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
