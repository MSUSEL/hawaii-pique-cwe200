/**
 * @name Hard-coded sensitive field
 * @description Hard-coding a sensitive string may compromise security.
 * @kind problem
 * @problem.severity error
 * @security-severity 9.8
 * @precision low
 * @id java/hardcoded-sensitive-field
 * @tags security
 * 		   external/cwe/cwe-540
 *       external/cwe/cwe-798
 */

 import java
 import semmle.code.java.security.HardcodedPasswordField
 
 from SensitiveVariable f, CompileTimeConstantExpr e
 where passwordFieldAssignedHardcodedValue(f, e)
 select f, "Sensitive field is assigned a hard-coded $@.", e, "value"