public class BAD_DifferentErrorCodes {
    public void checkStatus(String statusCode) {
        if (statusCode.equals("200")) {
            System.out.println("OK");
        } else {
            if (statusCode.equals("404")) {
                System.out.println("Error: Not Found");
            } else {
                if (statusCode.equals("500")) {
                    System.out.println("Error: Internal Server Error");
                } else {
                    System.out.println("Error: Unknown Status Code");
                }
            }
        }
    }
}
