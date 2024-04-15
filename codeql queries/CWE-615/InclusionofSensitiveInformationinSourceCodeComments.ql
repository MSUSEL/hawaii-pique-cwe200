/**
 * @name Comment contains sensitive information
 * @description Identifies comments that might inadvertently contain sensitive information, which could lead to security risks if exposed in source code.
 * @kind problem
 * @problem.severity warning
 * @id java/sensitive-comments/615
 * @tags security
 *       external/cwe/cwe-615
 * @cwe CWE-615
 */

import java
import semmle.code.java.security.SensitiveVariables

class HardCodedSensitiveComments extends Javadoc {
  HardCodedSensitiveComments() {
    exists(|
      this.getAChild().(JavadocText).getText().matches("%" + suspiciousComments() + "%")
    )
  }
}

from HardCodedSensitiveComments comment
select comment, " This comment may have hardcoded sensitive info"