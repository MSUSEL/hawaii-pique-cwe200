public class BAD_CloudServiceConfigExposure {
    public static void main(String[] args) {
        try {
            String cloudAccessToken = "SilentForest123";
            // Command to deploy an application to a cloud service with an access token exposed
            Runtime.getRuntime().exec(new String[]{"cloud", "deploy", "--access-token", cloudAccessToken, "my-cloud-app"});
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
