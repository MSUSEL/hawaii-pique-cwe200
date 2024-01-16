import java

from
  Method m, CatchClause cc, ExprStmt es, MethodAccess printlnCall,
  MethodAccess exceptionMethodAccess
where
  // Find catch clauses that handle general exceptions
  cc.getEnclosingCallable() = m and
  cc.getACaughtType().getASupertype*().hasName("Exception") and
  // Find expression statements within these catch clauses
  es.getEnclosingCallable() = cc.getEnclosingCallable() and
  printlnCall = es.getExpr() and
  // Check for printing of exception details like stack trace
  printlnCall.getMethod().hasName("println")
select m.getBody(), "Potential CWE-537: Java runtime error message containing sensitive information"
