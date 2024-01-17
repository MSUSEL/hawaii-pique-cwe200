import java

from
  Method m, CatchClause cc, ExprStmt es, MethodAccess printlnCall,
  MethodAccess exceptionMethodAccess, VarAccess varAccess, RefType caughtType
where
  // Check if the method contains a catch clause
  cc.getEnclosingCallable() = m and
  // Check if the catch clause catches an exception
  caughtType = cc.getACaughtType() and
  caughtType.getASupertype*().hasName("Exception") and
  // Check if the catch clause contains a call to System.err.println
  es.getEnclosingCallable() = cc.getEnclosingCallable() and
  printlnCall = es.getExpr() and
  printlnCall.getMethod().hasName("println") and
  (
    // Direct call to getStackTrace
    exceptionMethodAccess = printlnCall.getAnArgument().(MethodAccess) and
    exceptionMethodAccess.getMethod().hasName("getStackTrace") and
    exceptionMethodAccess.getQualifier().(VarAccess).getVariable().getType() = caughtType
  )
  or
  // Check if the caught exception itself is printed
  varAccess = printlnCall.getAnArgument().(VarAccess) and
  varAccess.getVariable().getType() = caughtType
select m.getBody(), "Potential CWE-537: Java runtime error message containing sensitive information"
