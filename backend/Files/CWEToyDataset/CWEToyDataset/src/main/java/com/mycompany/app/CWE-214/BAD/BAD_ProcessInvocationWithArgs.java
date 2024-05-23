public class BAD_ProcessInvocationWithArgs {
    // This example demonstrates invoking a process with sensitive information passed directly via command-line arguments.
    // Sensitive data such as passwords passed this way can be viewed by other users on the system using tools like 'ps'.

    public static void main(String[] args) {
        String pass = "helloWorld123";
        invokeSensitiveProcess(pass);
    }

    public static void invokeSensitiveProcess(String password) {
        String[] command = {"sh", "-c", "someCommand --password=" + password};
        try {
            // Constructing the command with the password passed as an argument
            Runtime.getRuntime().exec(command);
            System.out.println("Process invoked with sensitive command-line arguments.");
        } catch (Exception e) {
            // Insecure: Printing stack trace might expose sensitive information indirectly
            e.printStackTrace();
            System.err.println("An error occurred while invoking the process with this command: " + command);
            
        }
    }
}
