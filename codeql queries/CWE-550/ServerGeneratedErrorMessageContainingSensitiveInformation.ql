import java

from
  Call call, CatchClause cc, ExprStmt es, MethodAccess printlnCall,
  MethodAccess exceptionMethodAccess
where
  // Find calls to methods in the java.net package
  call.getCallee().getDeclaringType().hasQualifiedName("java.net", _) and
  // Find catch clauses in the methods where these calls occur
  cc.getEnclosingCallable() = call.getEnclosingCallable() and
  // Find expression statements within these catch clauses
  es.getEnclosingCallable() = cc.getEnclosingCallable() and
  printlnCall = es.getExpr() and
  printlnCall.getMethod().hasName("println")
/*
 * I would like to get more specific by ensuring that something sensative is
 * printed. This works for now, but I would still like to be more specific
 * if I can get it to work.
 */

// and
// Check if the println or log methods are being used with exception method calls
// exceptionMethodAccess = printlnCall.getAnArgument().(MethodAccess)
// This part is currently not returning any results but I would like to somehow check to see if the exception itself is being printed
// and
// (
//   exceptionMethodAccess.getMethod().hasName("getMessage") or
//   exceptionMethodAccess.getMethod().hasName("toString") or
//   exceptionMethodAccess.getMethod().hasName("getStackTrace")
// )
select cc.getBlock(), "Potential sensitive information exposure in network operation method"
