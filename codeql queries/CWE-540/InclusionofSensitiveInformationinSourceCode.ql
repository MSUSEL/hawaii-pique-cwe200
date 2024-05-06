/**
 * @name CWE-540: Inclusion of Sensitive Information in Source Code (CWE-540)
 * @description Identifies instances where sensitive information such as credentials may be hardcoded in the source code, potentially leading to security vulnerabilities.
 * @kind problem
 * @problem.severity error
 * @id java/sensitive-info-in-code/540
 * @tags security
 *       external/cwe/cwe-540
 * @cwe CWE-540
 */

import java
import SensitiveInfo.SensitiveInfo

from SensitiveStringLiteral ssl
select ssl, "CWE-540 violation: sensitive information in source code."