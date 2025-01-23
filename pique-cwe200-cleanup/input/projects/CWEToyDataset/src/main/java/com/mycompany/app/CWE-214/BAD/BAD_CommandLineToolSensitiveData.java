public class BAD_CommandLineToolSensitiveData {

    public static void main(String[] args) {
        String sensitiveFilePath = "/path/to/sensitive/data.txt";
        try {
            String command = "cat " + sensitiveFilePath;
            Runtime.getRuntime().exec(new String[]{"sh", "-c", command});
            System.out.println("Invoked command-line tool with sensitive file path.");
        } catch (Exception e) {
            System.err.println("An error occurred while invoking the command-line tool using " + sensitiveFilePath) ;
        }
    }
}
