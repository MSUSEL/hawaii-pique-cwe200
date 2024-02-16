public class BAD_ProcessInvocationWithArgs {
    // This example demonstrates invoking a process with sensitive information passed directly via command-line arguments.
    // Sensitive data such as passwords passed this way can be viewed by other users on the system using tools like 'ps'.

    public static void main(String[] args) {
        try {
            // Invoking a process that requires a password as a command-line argument
            String[] command = {"sh", "-c", "someCommand --password=secretPassword"};
            Runtime.getRuntime().exec(command);
            System.out.println("Process invoked with sensitive command-line arguments.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
