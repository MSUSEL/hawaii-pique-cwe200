import java.util.Map;
import java.util.Collection;


public class BAD_NegativeArraySizeException {
    public void createInventoryList(int arraySize, Map<String, String> medicalRecords) throws NegativeArraySizeException{
        
        if (arraySize < 0) {
            throw new NegativeArraySizeException("Connot save records for : " + medicalRecords.values());
        }
        else{
            String[] storage = new String[arraySize];
            Collection<String> records = medicalRecords.values();
            storage = records.toArray(storage);
        }
    }
}
