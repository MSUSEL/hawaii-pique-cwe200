/**
 * @name Exposure of information through shell error message
 * @description Exposing error messages from shell commands can lead to information disclosure.
 * @kind path-problem
 * @problem.severity warning
 * @id java/shell-error-exposure
 * @tags security
 *       external/cwe/cwe-535
 * @cwe CWE-535
 */

import java
import semmle.code.java.dataflow.TaintTracking

class ShellErrorExposureConfig extends TaintTracking::Configuration {
  ShellErrorExposureConfig() { this = "ShellErrorExposureConfig" }

  override predicate isSource(DataFlow::Node source) {
    exists(MethodAccess ma |
      // Captures getting the error stream from a process
      ma.getMethod().hasName("getErrorStream") and
      // Ensure the Process is the result of exec or start, indicating command execution
      (ma.getQualifier().(VarAccess).getVariable().getAnAssignedValue() instanceof MethodAccess and
       ma.getQualifier().(VarAccess).getVariable().getAnAssignedValue().(MethodAccess).getMethod().hasName("exec") or
       ma.getQualifier().(VarAccess).getVariable().getAnAssignedValue().(MethodAccess).getMethod().hasName("start")) and
      source.asExpr() = ma
    )
    or
    exists(MethodAccess exec |
      // Direct use of user input in command execution
      exec.getMethod().hasName("exec") and
      exec.getMethod().getDeclaringType().hasQualifiedName("java.lang", "Runtime") and
      source.asExpr() = exec
    )
  }

  override predicate isSink(DataFlow::Node sink) {
    // System.out.println or similar direct output methods as sinks
    (exists(MethodAccess println |
      println.getMethod().hasName("println") and
      println.getQualifier().(VarAccess).getVariable().getType() instanceof RefType and
      ((RefType)println.getQualifier().(VarAccess).getVariable().getType()).hasQualifiedName("java.io", "PrintStream") and
      sink.asExpr() = println.getAnArgument()) 
    )
    or
    exists(MethodAccess getMessage |
      getMessage.getMethod().hasName(["getMessage", "getStackTrace", "getStackTraceAsString", "printStackTrace"]) and
      getMessage.getMethod().getDeclaringType().getASupertype*().hasQualifiedName("java.lang", "Throwable") and
      sink.asExpr() = getMessage
    )
  }

  override predicate isSanitizer(DataFlow::Node node) {
    exists(MethodAccess ma |
      // Use regex matching to check if the method name contains 'sanitize', case-insensitive
      ma.getMethod().getName().toLowerCase().matches("%sanitize%") or
      ma.getMethod().getName().toLowerCase().matches("%encrypt%")
      and
      node.asExpr() = ma.getAnArgument()
    )
  }  
}





from ShellErrorExposureConfig config, DataFlow::PathNode source, DataFlow::PathNode sink
where config.hasFlowPath(source, sink)
select source, sink



// from ShellErrorExposureConfig config, DataFlow::PathNode source, DataFlow::PathNode sink
// where config.hasFlowPath(source, sink)
// select sink.getNode(), source, sink, "Potential CWE-535: Exposure of information through shell error message"

// import java

// from MethodAccess execCall, CatchClause cc, Stmt stmt, ReturnStmt returnStmt, Expr expr
// where
//   // Find instances of executing a command
//   execCall.getMethod().hasName("exec") and
//   execCall.getReceiverType().hasQualifiedName("java.lang", "Runtime") and
//   // Find catch clauses in the same method as the exec call
//   execCall.getEnclosingCallable() = cc.getEnclosingCallable() and
//   (
//     cc.getACaughtType().hasName("IOException") or
//     cc.getACaughtType().hasName("InterruptedException")
//   ) and
//   // Iterate through statements in the catch block
//   stmt = cc.getBlock().getAChild*() and
//   stmt instanceof ReturnStmt and
//   returnStmt = stmt and
//   // Now iterate through the return statement's children
//   stmt = returnStmt.getAChild*().(Stmt) and
//   expr instanceof MethodAccess and
//   // Ensure that expression is in the same method as the exec call
//   expr.getEnclosingCallable() = execCall.getEnclosingCallable() and
//   // Find calls to an Exception that exposes information
//   (
//   expr.(MethodAccess).getMethod().hasName(["getMessage", "getStackTrace", "getStackTraceAsString", "printStackTrace"])
//   ) and
//   expr.(MethodAccess).getQualifier().(VarAccess).getVariable().getType() instanceof RefType and
//   expr.(MethodAccess).getQualifier().(VarAccess).getVariable().getType().(RefType).hasQualifiedName("java.lang", "Exception")
// select expr, "Potential CWE-535: Exposure of information through shell error message"