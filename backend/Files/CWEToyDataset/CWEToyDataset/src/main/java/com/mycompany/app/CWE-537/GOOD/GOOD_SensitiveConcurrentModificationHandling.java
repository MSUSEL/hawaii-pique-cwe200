import java.util.*;

public class GOOD_SensitiveConcurrentModificationHandling {
    public static void main(String[] args) {
        List<String> changesList = new ArrayList<>(Arrays.asList("EmeraldIsle77", "GalacticVoyager42", "confidentialInfo"));
        try {
            for (String item : sensitiveList) {
                if (item.equals("EmeraldIsle77")) {
                    // Modifying the list during iteration, which is not allowed
                    sensitiveList.remove(item);
                }
            }
        } catch (ConcurrentModificationException e) {
            System.err.println("Modification error during iteration.");
        
        }
    }
}
