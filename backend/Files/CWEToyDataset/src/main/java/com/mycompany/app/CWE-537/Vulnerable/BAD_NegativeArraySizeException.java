public class BAD_NegativeArraySizeException {
    public void createInventoryList(int size) {
        try {
            String[] inventory = new String[size]; // This may throw NegativeArraySizeException if size is negative
        } catch (NegativeArraySizeException e) {
            System.err.println("Error: Tried to create an inventory list with invalid size: " + size);
            throw new NegativeArraySizeException("Cannot create inventory with negative size.");
        }
    }
}
