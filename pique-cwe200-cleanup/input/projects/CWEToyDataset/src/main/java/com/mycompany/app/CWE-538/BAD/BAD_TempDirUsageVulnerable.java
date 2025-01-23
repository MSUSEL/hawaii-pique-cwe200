import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

public class BAD_TempDirUsageVulnerable {
    void exampleVulnerable() throws IOException {
        File temp1 = File.createTempFile("random", ".txt"); 

        File temp2 = File.createTempFile("random", "file", null); 

        File systemTempDir = new File(System.getProperty("java.io.tmpdir"));
        File temp3 = File.createTempFile("random", "file", systemTempDir); 

        new File(System.getProperty("java.io.tmpdir"), "/child").mkdir(); 
        File tempDirChildFile = new File(System.getProperty("java.io.tmpdir"), "/child-create-file.txt");
        Files.createFile(tempDirChildFile.toPath());

        File tempDirChildDir = new File(System.getProperty("java.io.tmpdir"), "/child-dir");
        tempDirChildDir.mkdir(); 
        Files.createDirectory(tempDirChildDir.toPath()); 
    }
}
