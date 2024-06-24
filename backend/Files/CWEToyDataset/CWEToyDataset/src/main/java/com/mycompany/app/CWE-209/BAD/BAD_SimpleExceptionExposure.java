public class BAD_SimpleExceptionExposure {
    
    public void exposeException() {
    try {
        int a = 10/0;
        }
        catch (Exception e) {
        System.out.println(e);
        }
    }
    
}
