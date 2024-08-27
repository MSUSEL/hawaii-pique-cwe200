export const stringsPrompt = `
Sensitive strings are those that, if exposed, could lead to a vulnerability or make the system vulnerable.
Specify, I am interested in hardcoded string literals that may contain sensitive information. 

Be mindful of the context in which the string is used. 
For example, is it actually a password, API key, or other sensitive information? 
Also, make sure it's not dummy data or a placeholder used for testing purposes.

Here are some examples I would like to highlight:
1) "Code to handle sensitive personal information" - This is not a sensitive string. This is becasuse even though it says "sensitive personal information" it is not actually sensitive information. So, be mindful to not include strings that are not actually sensitive even if they talk about things that might be sensitive. If they don't expose sensitive information, they are not sensitive strings.
2) "https://api.example.com/data" - This is also not a sensitive string. It is instead dummy data as it is an example URL. This is likley used for testing purposes and is not sensitive information.
3) System.getenv("API_DEBUG") - This is not a sensitive string. This is because it is a key used to look up a value and not the value itself.
4) "password" - This is not a sensitive string. This is because it is a common word and is not actually a password.
5) String password = "secretPassword123"; - This is a sensitive string. This is because it is a password which is sensitive information.
6) String apiKey = "ABCD-1234-EFGH-5678"; - This is a sensitive string. This is because it is an API key which is sensitive information.
7) String ssn = "123-45-6789"; - This is a sensitive string. This is because it is a social security number which is sensitive information.
8) String creditCardNumber = "4111 1111 1111 1111"; - This is a sensitive string. This is because it is a credit card number which is sensitive information.
9) String dbConnection = "jdbc:mysql://user:password@localhost:3306/mydatabase"; - This is a sensitive string. This is because it is a database connection string which is sensitive information.
10) String encryptionKey = "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA"; - This is a sensitive string. This is because it is an encryption key which is sensitive information.

The main point is that you should be positive without a doubt that a string is sensitive to include it in your response. Don't get confused by common words or dummy data. Or by strings that talk about sensitive data Such as "with API Key:", but don't actually contain sensitive information.

### Guidelines to Reduce False Positives
1. **Context Matters:** Ensure that the string's context within the code indicates its sensitive nature. For example, "password" in a comment may not be sensitive, but in an assignment like String password = "example", it is.
2. **Common Words and Phrases:** Exclude common words and phrases unless they are in contexts indicating sensitivity (e.g., "admin" in username = "admin"; could be sensitive, but System.out.println("admin"); is not).
3. **Generic File Paths and URLs:** Exclude generic file paths and URLs that are unlikely to be sensitive, such as /home/user or http://example.com. Only include them if you are positive that they are sensitive as this contributes to many false positives.
4. **Review Surrounding Code:** Consider the surrounding code to determine if the string is part of a sensitive operation (e.g., database connection setup, user authentication process).
5. **Testing Words:** If a string has words like foo, bar, test, example, etc, it probably isn’t sensitive so don’t include them unless you are positive. 
6. **Confidence:** The main point is that you should be positive without a doubt that a string is sensitive to include it in your response. 


### Report Format
Provide a JSON response in the following format. Do not include any error messages or notes:
1. Provide a JSON response for each file that matches the format below.
    - The "name" field should be whole string just as it is given.
    - The "isSensitive" field should be "yes" if you think this is a sensitive string, and "no" if you don't think it is a sensitive string.
    - The "reason" field should describe the type of sensitive information found.
2. Do not include any non-printable or special characters in the name or description fields as this will cause the JSON to be invalid.
{
  "files": [
    {
      "fileName": "FileName1.java",
      "strings": [
        {
          "name": "string",
          "isSensitive": "yes/no",
          "reason": "reason for the decision"
        }
      ]
    }
  ]
}`

// ### Examples of Sensitive Strings
// These examples are not exhaustive; many other instances may still be considered sensitive.

// - String password = "secretPassword123";
// - String apiKey = "ABCD-1234-EFGH-5678";
// - String ssn = "123-45-6789";
// - String creditCardNumber = "4111 1111 1111 1111";
// - String dbConnection = "jdbc:mysql://user:password@localhost:3306/mydatabase";
// - String encryptionKey = "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA";

// ### Examples of Non-Sensitive Strings
// These examples are not exhaustive; many other instances may still be considered non-sensitive.

// - String welcomeMessage = "Welcome to our application!";
// - String filePath = "/home/user/documents";
// - String url = "http://example.com";
// - String placeholderText = "Enter your name";
// - String status = "Logged in successfully";
// - String errorMessage = "Invalid username or password";
// - String userName = "username";
// - String secret = "passphrase";
// - String password = "password";
// - alert('hello');

// Also, keys used to look up values aren't sensitive either from example:
// System.getenv("API_DEBUG")
// In this case "API_DEBUG" is not sensitive. 


// ### Guidelines to Reduce False Positives
// 1. **Context Matters:** Ensure that the string's context within the code indicates its sensitive nature. For example, "password" in a comment may not be sensitive, but in an assignment like String password = "example", it is.
// 2. **Common Words and Phrases:** Exclude common words and phrases unless they are in contexts indicating sensitivity (e.g., "admin" in username = "admin"; could be sensitive, but System.out.println("admin"); is not).
// 3. **Generic File Paths and URLs:** Exclude generic file paths and URLs that are unlikely to be sensitive, such as /home/user or http://example.com. Only include them if you are positive that they are sensitive as this contributes to many false positives.
// 4. **Review Surrounding Code:** Consider the surrounding code to determine if the string is part of a sensitive operation (e.g., database connection setup, user authentication process).
// 5. **Testing Words:** If a string has words like foo, bar, test, example, etc, it probably isn’t sensitive so don’t include them unless you are positive. 
// 6. **Confidence:** The main point is that you should be positive without a doubt that a string is sensitive to include it in your response. 
