/**
 * @name Potential CWE-536: Servlet Runtime Error Message Containing Sensitive Information
 * @description Identifies instances in servlets where sensitive error information might be exposed to the client, potentially leading to information disclosure.
 * @kind problem
 * @problem.severity warning
 * @id java/servlet-sensitive-error-exposure
 * @tags security
 *       external/cwe/cwe-536
 * @precision medium
 * 
 * @qlpack ql/java-all
 */

import java

// Find servlet methods that potentially expose sensitive error information
from Class servletClass, Method servletMethod, CatchClause cc, MethodAccess printlnCall, MethodAccess exceptionExposureCall, 
Expr printlnArg, AddExpr concatExpr


where
  // Look for classes that inherit from HttpServlet
  servletClass.getASupertype*().hasQualifiedName("javax.servlet.http", "HttpServlet") and
  servletMethod.getDeclaringType() = servletClass and
  cc.getEnclosingCallable() = servletMethod and
  (
    cc.getACaughtType().hasName("Exception") 
  ) 
  and 

  // Check if the sensitive informaiton is exposed via a print statement
  (
    printlnCall.getMethod().hasName("println") and
    printlnCall.getEnclosingCallable() = cc.getEnclosingCallable() and
    printlnCall.getEnclosingStmt() = cc.getBlock().getAChild*() and
    printlnArg = printlnCall.getAnArgument() and
    (
        // Direct call to Exposure
        (
            exceptionExposureCall = printlnArg and
            exceptionExposureCall instanceof MethodAccess and
            exceptionExposureCall.getMethod().hasName(["getMessage", "getStackTrace", "getStackTraceAsString", "printStackTrace"])
        ) or
        // Concatenation involving Exposure
        (
            printlnArg instanceof AddExpr and
            concatExpr = printlnArg and
            exceptionExposureCall = concatExpr.getAnOperand() and
            exceptionExposureCall instanceof MethodAccess and
            exceptionExposureCall.getMethod().hasName(["getMessage", "getStackTrace", "getStackTraceAsString", "printStackTrace"])
        ) and
        exceptionExposureCall.getQualifier() instanceof VarAccess and
        exceptionExposureCall.getQualifier().(VarAccess).getVariable().getType() instanceof RefType and
        exceptionExposureCall.getQualifier().(VarAccess).getVariable().getType().(RefType).hasQualifiedName("java.lang", "Exception")
    )
  )
 select exceptionExposureCall, "Potential CWE-536: Servlet Runtime Error Message Containing Sensitive Information"