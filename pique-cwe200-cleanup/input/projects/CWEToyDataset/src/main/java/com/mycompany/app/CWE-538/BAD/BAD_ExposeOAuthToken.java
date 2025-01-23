import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class BAD_ExposeOAuthToken {

    public static void main(String[] args) {
        String oauthToken = "abc123456789SECRET";
        try {
            // System temporary directory
            String tempDir = System.getProperty("java.io.tmpdir");

            // File to store sensitive OAuth token
            File tokenFile = new File(tempDir, "oauth_token.txt");

            // Write sensitive OAuth token
            try (FileWriter writer = new FileWriter(tokenFile)) {
                writer.write("oauthToken=" + oauthToken);
            }

            System.out.println("OAuth token written to: " + tokenFile.getAbsolutePath());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
