public class GOOD_FileDownloadAuthorization {
    public void authorizeDownload(String fileType, String userRole) {
        if (!fileType.isEmpty()) {
            if (userRole.equals("admin")) {
                System.out.println("Download Authorized");
            } else {
                System.out.println("Download Denied");
            }
        } else {
            System.out.println("Download Denied");
        }
    }
}
