import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class BAD_SensitivePathDisclosureInErrors{
    public static void main(String[] args) {
        String filePath = "/sensitive/config/taxes.txt";

        try {
            Scanner scanner = new Scanner(new File(filePath));
            while (scanner.hasNextLine()) {
                System.out.println(scanner.nextLine());
            }
            scanner.close();
        } catch (FileNotFoundException e) {
            throw new RuntimeException("Error: File not found - " + filePath);
        } catch (Exception e) {
            System.err.println(e.getMessage());
        }
    }
}
