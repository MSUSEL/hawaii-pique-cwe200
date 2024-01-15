package snippets;

public class JavaRuntimeErrorMessageContainingSensitiveInformation {
    public static void main(String[] args) {
        try {
            int result = 1 / 0; // Simulating a runtime error
            System.out.println("Result: " + result);
        } catch (Exception e) {
            // Exposing sensitive information in the error message
            System.out.println("Error: " + e.getStackTrace());
        }
    }
}
