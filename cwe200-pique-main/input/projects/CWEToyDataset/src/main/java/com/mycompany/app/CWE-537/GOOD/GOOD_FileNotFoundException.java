import java.io.*;

public class GOOD_FileNotFoundException {

    public void processFile(String filePath) {
        try {
            new FileReader(filePath);
        } catch (FileNotFoundException e) {
            System.err.println("File processing failed.");
            throw new RuntimeException("File not found");
        }
    }
}
