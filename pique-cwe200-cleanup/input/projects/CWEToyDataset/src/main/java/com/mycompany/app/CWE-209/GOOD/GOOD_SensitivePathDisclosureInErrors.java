import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class GOOD_SensitivePathDisclosureInErrors{

    public static void main(String[] args) {
        String filePath = "/sensitive/config/users.txt";

        try {
            Scanner scanner = new Scanner(new File(filePath));
            while (scanner.hasNextLine()) {
                System.out.println(scanner.nextLine());
            }
            scanner.close();
        } catch (FileNotFoundException e) {
            System.err.println("Error: File not found - Access denied or file does not exist.");
        } catch (Exception e) {
            System.err.println("An unexpected error occurred.");
        }
    }
}