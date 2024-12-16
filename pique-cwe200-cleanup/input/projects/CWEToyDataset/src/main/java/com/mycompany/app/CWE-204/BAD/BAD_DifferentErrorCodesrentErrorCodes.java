public class BAD_DifferentErrorCodesrentErrorCodes {
    public void checkStatus(String statusCode) {
        if (statusCode != null) {
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
        } else {
            System.out.println("Error");
        }

    }
}



