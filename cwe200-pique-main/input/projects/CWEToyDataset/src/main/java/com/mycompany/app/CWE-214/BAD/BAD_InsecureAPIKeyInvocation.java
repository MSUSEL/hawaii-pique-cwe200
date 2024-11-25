import java.io.IOException;

public class BAD_InsecureAPIKeyInvocation {

    public static void main(String[] args) {
        String apiKey = "MountainStream42";
        try {
            Runtime.getRuntime().exec("script --use-key " + apiKey);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
