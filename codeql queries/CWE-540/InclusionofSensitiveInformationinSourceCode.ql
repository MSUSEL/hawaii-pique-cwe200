/**
 * @name Inclusion of Sensitive Information in Source Code (CWE-540)
 * @description Identifies instances where sensitive information such as credentials may be hardcoded in the source code, potentially leading to security vulnerabilities.
 * @kind problem
 * @problem.severity error
 * @id CWE-540
 * @tags security
 *       external/cwe/cwe-540
 * @cwe CWE-540
 */

import java
import semmle.code.java.security.SensitiveVariables

from SensitiveStringLiteral ssl
select ssl, "Potential CWE-540 violation: sensitive information in source code."



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