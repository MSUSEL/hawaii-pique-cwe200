/**
 * @name Potential CWE-550: Server Generated Error Message Containing Sensitive Information
 * @description Identifies potential exposure of sensitive information through server-generated error messages that might be printed or logged within catch blocks during network operations, potentially leading to information disclosure.
 * @kind problem
 * @problem.severity warning
 * @id java/server-generated-sensitive-info-exposure
 * @tags security
 *       external/cwe/cwe-550
 * @precision medium
 * 
 * @qlpack ql/java-all
 */

import java

from
  Call call, CatchClause cc, ExprStmt es, MethodAccess printlnCall,
  MethodAccess sensitiveMethodCall , Expr printlnArg, AddExpr concatExpr
where
  // Find calls to methods in the java.net package
  call.getCallee().getDeclaringType().hasQualifiedName("java.net", _) and
  // Find catch clauses in the methods where these calls occur
  cc.getEnclosingCallable() = call.getEnclosingCallable() and
  
  // Check if the sensitive informaiton is exposed via a print statement
  (
    printlnCall.getMethod().hasName("println") and
    // Ensure that the print statement is in the catch clause
    printlnCall.getEnclosingCallable() = cc.getEnclosingCallable() and
    // Get all the statements in the catch clause
    printlnCall.getEnclosingStmt() = cc.getBlock().getAChild*() and
    // Get the argument to the print statement
    printlnArg = printlnCall.getAnArgument() 
    and
    (
        // Direct call to Exposure
        ( 
            sensitiveMethodCall  = printlnArg and
            sensitiveMethodCall  instanceof MethodAccess and
            sensitiveMethodCall .getMethod().hasName(["getMessage", "getStackTrace", "getStackTraceAsString", "printStackTrace"])
        ) or
        // Concatenation involving Exposure
        (
            printlnArg instanceof AddExpr and
            concatExpr = printlnArg and
            sensitiveMethodCall  = concatExpr.getAnOperand() and
            sensitiveMethodCall  instanceof MethodAccess and
            sensitiveMethodCall .getMethod().hasName(["getMessage", "getStackTrace", "getStackTraceAsString", "printStackTrace"])
        ) and
        sensitiveMethodCall .getQualifier() instanceof VarAccess and
        sensitiveMethodCall .getQualifier().(VarAccess).getVariable().getType() instanceof RefType
    )
  )

  select sensitiveMethodCall , "Potential CWE-550: Server Generated Error Message Containing Sensitive Information"