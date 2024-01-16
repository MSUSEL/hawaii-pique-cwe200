import java

from MethodAccess execCall, CatchClause cc, Stmt stmt, Expr expr, MethodAccess methodCall
where
  // Find instances of executing a command
  execCall.getMethod().hasName("exec") and
  execCall.getReceiverType().hasQualifiedName("java.lang", "Runtime") and
  // Find catch clauses in the same method as the exec call
  execCall.getEnclosingCallable() = cc.getEnclosingCallable() and
  (
    cc.getACaughtType().hasName("IOException") or
    cc.getACaughtType().hasName("InterruptedException")
  ) and
  // Iterate through statements in the catch block
  stmt = cc.getBlock().getAChild*()
select stmt, "Potential CWE-535: Exposure of information through shell error message"
