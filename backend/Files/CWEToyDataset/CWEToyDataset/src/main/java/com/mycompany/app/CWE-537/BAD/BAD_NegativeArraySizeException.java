public class BAD_NegativeArraySizeException {
    public void createInventoryList(int secretArraySize) {
        try {
            String[] inventory = new String[secretArraySize]; // This may throw NegativeArraySizeException if size is negative
        } catch (NegativeArraySizeException e) {
            System.err.println("Error: Tried to create an inventory list with invalid size: " + secretArraySize);
            throw new NegativeArraySizeException("Cannot create inventory with invalid size " + secretArraySize);
        }
    }
}
