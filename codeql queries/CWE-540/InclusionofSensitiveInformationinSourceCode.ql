import java
import semmle.code.java.security.SensitiveVariables


private string suspicious() {
    result =
      [
          "username","password","userId","userDetails","username","password","dbUrl","dbUser","dbPassword","DB_USER","DB_PASSWORD","DB_URL","creditCardNumber","payload","prop","username","password","smtpHost","smtpPort","sensitiveContent","username","password","KEY","APIKey","sensitiveData","validUsername","VALID_USER","USER_PASSWORD","userData","username","password","VALID_USERNAME","VALID_PASSWORD","ATTEMPT_USERNAME","ATTEMPT_PASSWORD","username","password","validUsername","validPassword","VALID_USERNAME","VALID_PASSWORD","VALID_USERNAME","VALID_PASSWORD","SECRET_TOKEN","inputPassword","actualPassword","username","jdbc:mysql://localhost:3306/appdb","user","password","inputPassword","actualPassword","query","filePath","username","password","username","password","query","configPath","filePath","connection","user","pass","sanitizedMessage","accountNumber","amount","uniqueErrorCode","conn","username","File","endpointUrl","conn","File","aggregatedData","connection","username","password","userInput","sensitiveFilePath","API_KEY","pass","secretPassword","username","password","username","password","creditCardNumber","userId","username","password","req","resp","DEBUG_MODE","logger","prop","logger","sensitiveInfo","logger","req","resp","apiKey","personalEmail","personalPhone","username","password","username","password","apiKey","dbUrl","user","password","creditCardNumber","expiryDate","cvv","username","password","username","password","scriptPath","command","command","command","errorOutput","command","conn","url","connection","filename","file","filename","file","url","user","password","filePath","address","secureDataFetch","performSensitiveOperation","username","password","fullName","creditCardNumber","TRANSACTION_LOG_FILE","dbUrl","dbUser","dbPassword","dbUrl","username","SERVICE_API_KEY","DATABASE_URL","DATABASE_USER","DATABASE_PASSWORD","apiKey","encryptionKey","databaseUrl","databaseUser","databasePassword","databaseUrl","databaseUser","databasePassword","documentPath","path","directoryPath","requestedFileName","fileDirectory","tempFilesPath","e","connection","e","e","e","cardNumber","expiryDate","cvv","patientId","symptoms","doctorId","username","oldPassword","newPassword","encryptionKey"
      ]
  }

from SensitiveStringLiteral ssl
select ssl, "Potential CWE-540 violation: sensitive information $@. in source code."



// class SensitiveStringLiteral extends StringLiteral {
//     SensitiveStringLiteral() {
//       // Check for matches against the suspicious patterns
//       this.getValue().regexpMatch(suspicious()) and
//       not exists(MethodAccess ma |
//         ma.getAnArgument() = this and
//         (
//           ma.getMethod().hasName("getenv") or
//           ma.getMethod().hasName("getParameter") or
//           ma.getMethod().hasName("getProperty") 
//         )
//       )
//     }   
// }