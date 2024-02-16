public class BAD_CommandLineToolSensitiveData {
    // Demonstrates the risky practice of using sensitive data directly in command-line tool invocations,
    // which could be exposed to other system users through process listings.

    public static void main(String[] args) {
        String sensitiveFilePath = "/path/to/sensitive/data.txt";
        // Invoking a command-line tool with a sensitive file path as an argument
        try {
            String command = "cat " + sensitiveFilePath; // Example command that prints file content
            Runtime.getRuntime().exec(new String[]{"sh", "-c", command});
            System.out.println("Invoked command-line tool with sensitive file path.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
