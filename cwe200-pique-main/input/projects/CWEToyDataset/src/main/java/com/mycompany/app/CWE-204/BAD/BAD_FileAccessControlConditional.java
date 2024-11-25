public class BAD_FileAccessControlConditional {
    public void accessFile(String userRole, String filePath) {
        if (userRole.equals("admin")) {
            System.out.println("Access Granted: Admin has full access");
        } else if (userRole.equals("user")) {
            if (filePath.startsWith("/public/")) {
                System.out.println("Access Granted: User can access public files");
            } else {
                System.out.println("Access Denied: User cannot access private files");
            }
        } else {
            System.out.println("Access Denied: Unknown role");
        }
    }
}
