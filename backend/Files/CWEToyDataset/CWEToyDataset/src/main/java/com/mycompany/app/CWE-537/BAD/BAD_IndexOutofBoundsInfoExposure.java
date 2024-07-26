public class BAD_IndexOutofBoundsInfoExposure {

    public static void main(String[] args, int index) throws IndexOutOfBoundsException{
        String[] dataMap = {"MistyMountain2020", "medicalRecords"};
        
        if (index < 0 || index >= sensitiveData.length) {
            throw new IndexOutOfBoundsException("Index " + index + " is out of bounds for the array" + dataMap) ;
        }
        else{
            String data = dataMap[index]; // This will throw IndexOutOfBoundsException
        }
    }
}
