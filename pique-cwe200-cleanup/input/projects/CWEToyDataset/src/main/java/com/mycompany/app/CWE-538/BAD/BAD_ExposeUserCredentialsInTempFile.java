import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class BAD_ExposeUserCredentialsInTempFile {

    public static void main(String[] args) {
        String password = "admin123";
        try {
            // Access the temporary directory
            String tempDir = System.getProperty("java.io.tmpdir");

            // Create a credentials file
            File credentialsFile = new File(tempDir, "user_credentials.txt");

            // Write sensitive user credentials
            try (FileWriter writer = new FileWriter(credentialsFile)) {
                writer.write("username=admin\npassword=" + password);
            }

            System.out.println("User credentials written to: " + credentialsFile.getAbsolutePath());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
