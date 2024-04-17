import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class BAD_IndexOutofBoundsInfoExposure {

    private static final Logger logger = LogManager.getLogger(BAD_IndexOutofBoundsInfoExposure.class);

    public static void main(String[] args) {
        String[] sensitiveData = {"secretPassword123", "confidentialInfo"};
        try {
            // Attempting to access an index that does not exist
            String data = sensitiveData[10]; // This will throw IndexOutOfBoundsException
        } catch (IndexOutOfBoundsException e) {
            // Inappropriate handling of IndexOutOfBoundsException using Log4j
            logger.error("Caught an exception while accessing data: ", e);
            // Exposing potentially sensitive information about the array bounds
            logger.error("This array should only be accessed within its bounds. Example sensitive data: {}", sensitiveData[0]);
        }
    }
}
