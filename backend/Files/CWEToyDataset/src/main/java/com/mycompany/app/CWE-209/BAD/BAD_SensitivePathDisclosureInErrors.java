import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class BAD_SensitivePathDisclosureInErrors{
    // This example shows an insecure practice of including detailed file system paths in error messages
    // when file access is denied or the file does not exist.

    public static void main(String[] args) {
        String filePath = "/sensitive/config/secrets.txt";

        try {
            Scanner scanner = new Scanner(new File(filePath));
            while (scanner.hasNextLine()) {
                System.out.println(scanner.nextLine());
            }
            scanner.close();
        } catch (FileNotFoundException e) {
            // Insecure: Directly exposing the file path in the error message.
            System.err.println("Error: File not found - " + filePath);
        } catch (Exception e) {
            System.err.println("An unexpected error occurred.");
        }
    }
}
