public class BAD_PrintStackTrace {
    public static void main(String[] args) {
        String accountNumber = "928-1223-1221-1221";

        try {
            processInfo(accountNumber);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void processInfo(String accountNumber) throws Exception {
        if (accountNumber != null) {
            throw new Exception("Sensitive information: " + accountNumber);
        }
    }
}
