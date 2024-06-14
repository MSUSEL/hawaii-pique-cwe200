public class GOOD_IndexOutofBoundsInfoExposure {

    public static void main(String[] args) {
        String[] sensitiveData = {"secretPassword123", "confidentialInfo"};
        try {
            // Attempting to access an index that does not exist
            String data = sensitiveData[10]; // This will throw IndexOutOfBoundsException
        } catch (IndexOutOfBoundsException e) {
            // Inappropriate handling of IndexOutOfBoundsException using Log4j
            System.err.println("Caught an exception while accessing data.");
            // Exposing potentially sensitive information about the array bounds
            System.err.println("This array should only be accessed within its bounds.");
        }
    }
}
