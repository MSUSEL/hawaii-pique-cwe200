public class BAD_IndexOutofBoundsInfoExposure {

    public static void main(String[] args) {
        String[] sensitiveData = {"secretPassword123", "confidentialInfo"};
        try {
            // Attempting to access an index that does not exist
            String data = sensitiveData[10]; // This will throw IndexOutOfBoundsException
        } catch (IndexOutOfBoundsException e) {
            System.err.println("This array should only be accessed within its bounds. Example sensitive data: " + sensitiveData[0]);
            throw new IndexOutOfBoundsException("This array should only be accessed within its bounds. Example sensitive data: " + sensitiveData[0]);
        }
    }
}
