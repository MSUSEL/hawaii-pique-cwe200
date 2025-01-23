public class BAD_ProcessInvocationWithArgs {

    public static void main(String[] args) {
        String pass = "helloWorld123";
        invokeSensitiveProcess(pass);
    }

    public static void invokeSensitiveProcess(String password) {
        String[] command = {"sh", "-c", "someCommand --password=" + password};
        try {
            Runtime.getRuntime().exec(command);
            System.out.println("Process invoked with sensitive command-line arguments.");
        } catch (Exception e) {
            e.printStackTrace();
            System.err.println("An error occurred while invoking the process with this command: " + command);
            
        }
    }
}
