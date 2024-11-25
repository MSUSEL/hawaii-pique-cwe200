public class GOOD_SensitiveTypeNotPresentExceptionHandling {
    public static void main(String[] args) {
        try {
            // Attempting to access a type by its name which might not be present
            Class<?> sensitiveClass = Class.forName("com.secret.config.SensitiveConfiguration");
        } catch (TypeNotPresentException e) {
            System.err.println("Failed to find required class");
        } catch (ClassNotFoundException e) {
            System.err.println("Class not found");
        }
    }
}
