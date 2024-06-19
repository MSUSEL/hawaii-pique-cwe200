import java.io.*;

public class BAD_FileNotFoundException {

    public void processFile(String filePath) throws FileNotFoundException{

        if (!isValudFilePath(filePath)) {
            throw new FileNotFoundException("File not found at " + filePath);
        }
        else{
            FileReader reader = new FileReader(filePath);
            reader.close();
        }
    }

    private boolean isValudFilePath(String filePath) {
        // Check if the file path is valid
        return true;
    }
}
