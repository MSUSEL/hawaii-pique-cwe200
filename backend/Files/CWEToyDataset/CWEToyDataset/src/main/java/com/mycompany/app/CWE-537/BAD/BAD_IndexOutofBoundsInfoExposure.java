public class BAD_IndexOutofBoundsInfoExposure {

    public static void main(String[] args, int index) throws IndexOutOfBoundsException{
        String[] sensitiveData = {"secretPassword123", "confidentialInfo"};
        
        if (index < 0 || index >= sensitiveData.length) {
            throw new IndexOutOfBoundsException("Index " + index + " is out of bounds for the array" + sensitiveData) ;
        }
        else{
            String data = sensitiveData[index]; // This will throw IndexOutOfBoundsException
        }
    }
}
