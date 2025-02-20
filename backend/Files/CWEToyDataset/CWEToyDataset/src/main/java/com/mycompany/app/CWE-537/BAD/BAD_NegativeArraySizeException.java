import java.util.Map;
import java.util.Collection;

public class BAD_NegativeArraySizeException {
    public void createInventoryList(int arraySize, Map<String, String> medicalRecords) {
        try {
            if (arraySize < 0) {
                throw new NegativeArraySizeException("Cannot save records for: " + medicalRecords);
            } else {
                String[] storage = new String[arraySize];
                Collection<String> records = medicalRecords.values();
                storage = records.toArray(storage);
                System.out.println("Records saved successfully.");
            }
        } catch (NegativeArraySizeException e) {
            System.err.println(e.getMessage());
        }
    }

}
