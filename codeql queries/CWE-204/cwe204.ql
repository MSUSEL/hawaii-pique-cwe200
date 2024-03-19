/**
 * @name Observable discrepancies in sensitive error messages
 * @description Detects if statements within sensitive contexts that produce different error messages based on conditional branches, which could lead to observable discrepancies.
 * @kind problem
 * @problem.severity warning
 * @id CWE-204
 * @tags security
 *       external/cwe/cwe-204
 * @cwe CWE-204
 */

 import java
 import semmle.code.java.dataflow.FlowSources
 import semmle.code.java.security.SensitiveActions
 import semmle.code.java.controlflow.Guards
 
 // Class for String literals potentially used in observable discrepancies
 class SensitiveMessageLiteral extends StringLiteral {
   SensitiveMessageLiteral() {
     this.getValue().regexpMatch(".*Login (Successful|Failed).*|.*Invalid (username|password).*|.*Error.*") // Adjust the regex to match relevant messages
   }
 }
 
 from IfStmt outerIf, IfStmt innerIf, SensitiveMessageLiteral str1, SensitiveMessageLiteral str2, SensitiveMessageLiteral str3
 where
   // Check if the innerIf is directly within the body of outerIf
   outerIf.getAChild*() = innerIf and
   // Check for specific message literals in the then and else branches of the inner if-statement and the else branch of the outer if-statement
   innerIf.getThen().getBasicBlock().getANode() = str1 and
   innerIf.getElse().getBasicBlock().getANode() = str2 and
   outerIf.getElse().getBasicBlock().getANode() = str3 and
   // Ensure the innerIf and outerIf are not the same
   outerIf != innerIf
 select outerIf.getBasicBlock(), "This 'if' statement in a sensitive method has different messages in its branches, potentially indicating observable discrepancies (CWE-204)."
 