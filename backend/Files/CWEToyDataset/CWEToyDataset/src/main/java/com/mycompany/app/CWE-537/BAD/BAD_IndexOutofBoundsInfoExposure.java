public class BAD_IndexOutofBoundsInfoExposure {

    public static void main(String[] args, int index) throws IndexOutOfBoundsException{
        String[] dataMap = {"MistyMountain2020", "medicalRecords"};
        
        if (index < 0 || index >= dataMap.length) {
            throw new IndexOutOfBoundsException("Index " + index + " is out of bounds for the array" + dataMap[index]); 
        }
        else{
            String data = dataMap[index];
        }
    }
}
