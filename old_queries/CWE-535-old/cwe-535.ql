/**
 * @name Potential CWE-535: Exposure of information through shell error message
 * @description Identifies potential exposure of sensitive information through error messages resulting from executing shell commands and capturing their output or errors in Java applications.
 * @kind problem
 * @problem.severity warning
 * @id java/potential-cwe-535-shell-error-exposure
 * @tags security
 *       external/cwe/cwe-535
 * @precision medium
 * 
 * @qlpack ql/java-all
 */

import java

from MethodAccess execCall, CatchClause cc, Stmt stmt, ReturnStmt returnStmt, Expr expr
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
  stmt = cc.getBlock().getAChild*() and
  stmt instanceof ReturnStmt and
  returnStmt = stmt and
  // Now iterate through the return statement's children
  stmt = returnStmt.getAChild*().(Stmt) and
  expr instanceof MethodAccess and
  // Ensure that expression is in the same method as the exec call
  expr.getEnclosingCallable() = execCall.getEnclosingCallable() and
  // Find calls to an Exception that exposes information
  (
  expr.(MethodAccess).getMethod().hasName(["getMessage", "getStackTrace", "getStackTraceAsString", "printStackTrace"])
  ) and
  expr.(MethodAccess).getQualifier().(VarAccess).getVariable().getType() instanceof RefType and
  expr.(MethodAccess).getQualifier().(VarAccess).getVariable().getType().(RefType).hasQualifiedName("java.lang", "Exception")
select expr, "Potential CWE-535: Exposure of information through shell error message"