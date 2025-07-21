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
 
 from string pattern, string fileName, Javadoc comment
 where sensitiveComments(fileName, pattern)
   and comment.getFile().getBaseName() = fileName
   and comment.getAChild().(JavadocText).getText().trim().matches(pattern.trim())
 select comment, "Sensitive info detected: " + pattern
 