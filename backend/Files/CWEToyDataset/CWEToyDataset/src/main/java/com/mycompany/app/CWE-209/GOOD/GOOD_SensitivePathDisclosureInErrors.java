import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class GOOD_SensitivePathDisclosureInErrors{
    // This example shows a secure practice of not including detailed file system paths in error messages
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
            // Secure: Avoiding direct exposure of the file path in the error message.
            System.err.println("Error: File not found - Access denied or file does not exist.");
        } catch (Exception e) {
            System.err.println("An unexpected error occurred.");
        }
    }
}