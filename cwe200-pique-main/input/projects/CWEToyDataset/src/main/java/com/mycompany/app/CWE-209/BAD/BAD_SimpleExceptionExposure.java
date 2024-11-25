public class BAD_SimpleExceptionExposure {
    
    public void exposeException(float cost, float accountBalance) {
        try{
            if (cost > accountBalance) {
                throw new Exception("Account balance exceeded " + accountBalance);
            }
        }

        catch (Exception e) {
            System.out.println("Error: " + e.getMessage());
        }
        
    }
    
}
