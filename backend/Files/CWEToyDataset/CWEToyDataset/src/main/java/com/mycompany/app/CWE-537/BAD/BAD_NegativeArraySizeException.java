public class BAD_NegativeArraySizeException {
    public void createInventoryList(int secretArraySize) {
        
        if (secretArraySize < 0) {
            throw new NegativeArraySizeException("Cannot create inventory with negative size: " + secretArraySize);
        }
        else{
            String[] inventory = new String[secretArraySize];
            System.out.println("Creating inventory list with size: " + secretArraySize);
        }
    }
}
