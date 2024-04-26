import java.util.*;

public class GOOD_SensitiveConcurrentModificationHandling {
    public static void main(String[] args) {
        List<String> sensitiveList = new ArrayList<>(Arrays.asList("secret1", "secret2", "confidentialInfo"));
        try {
            for (String item : sensitiveList) {
                if (item.equals("secret1")) {
                    // Modifying the list during iteration, which is not allowed
                    sensitiveList.remove(item);
                }
            }
        } catch (ConcurrentModificationException e) {
            // Inappropriately handling ConcurrentModificationException
            System.err.println("Modification error during iteration.");
        
        }
    }
}
