import java.io.*;

public class BAD_FileNotFoundException {

    public void processFile(String filePath) {
        try {
            FileReader reader = new FileReader(filePath);
            reader.close();
        } catch (IOException e) {
            // Handle or log the IOException
            System.err.println("An I/O error occurred while trying to access " + filePath);
        }
        
    }

}
