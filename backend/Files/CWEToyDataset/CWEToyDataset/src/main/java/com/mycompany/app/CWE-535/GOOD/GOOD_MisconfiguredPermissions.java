public class GOOD_MisconfiguredPermissions {

    public static void main(String[] args) {
        try {
            String scriptPath = "/usr/local/scripts/script.sh";
            ProcessBuilder pb = new ProcessBuilder("bash", scriptPath);
            Process process = pb.start();
            process.waitFor();

            System.out.println("Error.");
        } catch (Exception e) {
            System.err.println("An error has occured");        
        }
    }
}
