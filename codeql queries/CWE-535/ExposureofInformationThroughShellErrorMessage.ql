/**
 * @name Exposure of information through shell error message
 * @description Exposing error messages from shell commands can lead to information disclosure.
 * @kind path-problem
 * @problem.severity warning
 * @id CWE-535
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
    or
    exists(MethodAccess log |
      log.getMethod().getDeclaringType().hasQualifiedName("org.apache.logging.log4j", "Logger") and
      log.getMethod().hasName(["error", "warn", "info", "debug", "fatal"]) and
      sink.asExpr() = log.getAnArgument()
   )
    or
    exists(MethodAccess log |
      log.getMethod().getDeclaringType().hasQualifiedName("org.slf4j", "Logger") and
      log.getMethod().hasName(["error", "warn", "info", "debug"]) and
      sink.asExpr() = log.getAnArgument()
    )
  }

  override predicate isSanitizer(DataFlow::Node node) {
    exists(MethodAccess ma |
      // Use regex matching to check if the method name contains 'sanitize', case-insensitive
      (ma.getMethod().getName().toLowerCase().matches("%sanitize%") or
      ma.getMethod().getName().toLowerCase().matches("%encrypt%"))
      and
      node.asExpr() = ma.getAnArgument()
    )
  }  
}

from ShellErrorExposureConfig config, DataFlow::PathNode source, DataFlow::PathNode sink
where config.hasFlowPath(source, sink)
select sink.getNode(), source, sink, "Potential CWE-536: Servlet Runtime Error Message Containing Sensitive Information"