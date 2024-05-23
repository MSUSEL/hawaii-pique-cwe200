import java.io.*;
import java.util.logging.*;

public class GOOD_FileNotFoundException {
    private static final Logger logger = Logger.getLogger("FileProcessingLogger");

    public void processFile(String filePath) {
        try {
            new FileReader(filePath);
        } catch (FileNotFoundException e) {
            logger.warning("File processing failed.");
            throw new RuntimeException("File not found");
        }
    }
}
