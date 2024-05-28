public class GOOD_DifferentErrorCodes {
    public void checkStatus(String statusCode) {
        if (statusCode != null) {
            if (statusCode.equals("200")) {
                System.out.println("OK");
            } else {
                if (statusCode.equals("404")) {
                    System.out.println("Error");
                } else {
                    if (statusCode.equals("500")) {
                        System.out.println("Error");
                    } else {
                        System.out.println("Error");
                    }
                }
            }
        } else {
            System.out.println("Error");
        }

    }
}



