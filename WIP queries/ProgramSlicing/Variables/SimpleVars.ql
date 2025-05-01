/**
 * @name Find all SensitiveVariableExpr instances
 * @description Identifies all variables that are considered SensitiveVariableExpr.
 * @kind problem
 * @problem.severity warning
 * @id java/find-sensitive-variable-expr
 */

 import java
 import SensitiveInfo.SensitiveInfo
 
 from SensitiveVariableExpr sve, Variable v
 where sve = v.getAnAccess()
 select sve, v.getName().toString() +"|"+ v.getType().toString()