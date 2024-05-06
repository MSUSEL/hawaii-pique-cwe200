/**
 * @name CWE-615: Inclusion of Sensitive Information in Source Code Comments
 * @description Identifies comments that might inadvertently contain sensitive information, which could lead to security risks if exposed in source code.
 * @kind problem
 * @problem.severity warning
 * @id java/sensitive-comments/615
 * @tags security
 *       external/cwe/cwe-615
 * @cwe CWE-615
 */

import java
import SensitiveInfo.SensitiveInfo

class HardCodedSensitiveComments extends Javadoc {
  HardCodedSensitiveComments() {
    exists(string pattern |
      sensitiveComments(pattern) and
      this.getAChild().(JavadocText).getText().regexpMatch(".*" + pattern + ".*")
    )
  }
}

from HardCodedSensitiveComments comment
select comment, " This comment may have hardcoded sensitive info"