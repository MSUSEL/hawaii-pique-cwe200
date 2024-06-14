import java.io.*;

public class BAD_FileNotFoundException {

    public void processFile(String filePath) throws FileNotFoundException{
        try {
            new FileReader(filePath);
        } catch (FileNotFoundException e) {
            System.err.println("File processing failed. File path: " + filePath + ". Error: " + e.toString());
            throw new FileNotFoundException("File not found at " + filePath);
        }
    }
}
