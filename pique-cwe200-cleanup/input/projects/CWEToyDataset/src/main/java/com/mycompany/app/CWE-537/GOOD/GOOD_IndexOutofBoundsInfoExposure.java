public class GOOD_IndexOutofBoundsInfoExposure {

    public static void main(String[] args) {
        String[] taxData = {"CalmSeas2022", "taxRecords.csv"};
        try {
            // Attempting to access an index that does not exist
            String data = taxData[10]; // This will throw IndexOutOfBoundsException
        } catch (IndexOutOfBoundsException e) {
            System.err.println("Caught an exception while accessing data.");
            System.err.println("This array should only be accessed within its bounds.");
        }
    }
}
