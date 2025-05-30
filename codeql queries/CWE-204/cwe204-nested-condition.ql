/**
 * @name CWE-204: Observable discrepancies in sensitive error messages
 * @description Detects if statements within sensitive contexts that produce different error messages based on conditional branches, which could lead to observable discrepancies.
 * @kind problem
 * @problem.severity warning
 * @id java/error-message-discrepancies/204
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
     this.getValue().regexpMatch(".*(Login Successful|Invalid password|Invalid username|Access (Granted|Denied)|Verification (Successful|Failed)|Authentication (Successful|Failed)|User not found|Password cannot be empty|Username cannot be empty|Input cannot be null|Input cannot be empty|'admin' is a reserved keyword|Verification Successful: Email found in system|Verification Failed: Email not registered|Access Granted: Admin has full access|Access Granted: User can access public files|Access Denied: User cannot access private files|Access Denied: Unknown role|Download Authorized|Download Denied: Insufficient privileges|Authentication Successful: Device recognized|Authentication Failed: Device not recognized in local network|Authentication Failed: Unknown device).*")
   }
 }
 
 from IfStmt outerIf, IfStmt innerIf, SensitiveMessageLiteral innerVal, SensitiveMessageLiteral outerVal
 where
   // Check if the innerIf is directly within the body of outerIf
   outerIf.getAChild*() = innerIf and
   // Check for specific message literals in the then and else branches of the inner if-statement and the else branch of the outer if-statement
   innerIf.getElse().getBasicBlock().getANode() = innerVal and
   outerIf.getElse().getBasicBlock().getANode() = outerVal and
   // Ensure the innerIf and outerIf are not the same
   innerVal.getValue() != outerVal.getValue()
   
 select outerIf.getBasicBlock(), "CWE-204: Observable discrepancies due to different error messages in nested if-statements."