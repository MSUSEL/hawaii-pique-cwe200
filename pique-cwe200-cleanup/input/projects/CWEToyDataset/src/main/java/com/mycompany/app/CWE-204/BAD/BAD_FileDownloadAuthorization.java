public class BAD_FileDownloadAuthorization {
    public void authorizeDownload(String fileType, String userRole) {
        if (fileType.equals("confidential")) {
            if (userRole.equals("admin")) {
                System.out.println("Download Authorized");
            } else {
                System.out.println("Download Denied: Insufficient privileges");
            }
        } else {
            System.out.println("Download Authorized");
        }
    }
}
