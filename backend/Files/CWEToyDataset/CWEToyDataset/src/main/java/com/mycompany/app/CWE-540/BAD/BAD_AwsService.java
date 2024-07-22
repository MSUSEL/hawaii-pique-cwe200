public class BAD_AwsService {
    private static final String AWS_SECRET_ACCESS_KEY = "s3cr3tAcc355K3yV4lu3"; // Sensitive AWS secret key

    public void accessAwsService() {
        System.out.println("Accessing AWS with secret key: " + AWS_SECRET_ACCESS_KEY);
    }
}
