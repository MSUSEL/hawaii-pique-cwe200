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

module Flow = TaintTracking::Global<ShellErrorExposureConfig>;
import Flow::PathGraph


module ShellErrorExposureConfig implements DataFlow::ConfigSig {
 
  predicate isSource(DataFlow::Node source) {
    exists(MethodCall mc |
      // Captures getting the error stream from a process
      mc.getMethod().hasName("getErrorStream") and
      // Ensure the Process is the result of exec or start, indicating command execution
      (mc.getQualifier().(VarAccess).getVariable().getAnAssignedValue() instanceof MethodCall and
       mc.getQualifier().(VarAccess).getVariable().getAnAssignedValue().(MethodCall).getMethod().hasName("exec") or
       mc.getQualifier().(VarAccess).getVariable().getAnAssignedValue().(MethodCall).getMethod().hasName("start")) and
      source.asExpr() = mc
    )
    or
    exists(MethodCall exec |
      // Direct use of user input in command execution
      exec.getMethod().hasName("exec") and
      exec.getMethod().getDeclaringType().hasQualifiedName("java.lang", "Runtime") and
      source.asExpr() = exec
    )
  }

  predicate isSink(DataFlow::Node sink) {
    // System.out.println or similar direct output methods as sinks
    (exists(MethodCall println |
      println.getMethod().hasName("println") and
      println.getQualifier().(VarAccess).getVariable().getType() instanceof RefType and
      ((RefType)println.getQualifier().(VarAccess).getVariable().getType()).hasQualifiedName("java.io", "PrintStream") and
      sink.asExpr() = println.getAnArgument()) 
    )
    or
    exists(MethodCall getMessage |
      getMessage.getMethod().hasName(["getMessage", "getStackTrace", "getStackTraceAsString", "printStackTrace"]) and
      getMessage.getMethod().getDeclaringType().getASupertype*().hasQualifiedName("java.lang", "Throwable") and
      sink.asExpr() = getMessage
    )
    or
    exists(MethodCall log |
      log.getMethod().getDeclaringType().hasQualifiedName("org.apache.logging.log4j", "Logger") and
      log.getMethod().hasName(["error", "warn", "info", "debug", "fatal"]) and
      sink.asExpr() = log.getAnArgument()
   )
    or
    exists(MethodCall log |
      log.getMethod().getDeclaringType().hasQualifiedName("org.slf4j", "Logger") and
      log.getMethod().hasName(["error", "warn", "info", "debug"]) and
      sink.asExpr() = log.getAnArgument()
    )
  }

  predicate isBarrier(DataFlow::Node node) {
    exists(MethodCall mc |
      // Use regex matching to check if the method name contains 'sanitize', case-insensitive
      (mc.getMethod().getName().toLowerCase().matches("%sanitize%") or
      mc.getMethod().getName().toLowerCase().matches("%encrypt%") 
      )
      and
      node.asExpr() = mc.getAnArgument()
    )
  }   
}

from Flow::PathNode source, Flow::PathNode sink
where Flow::flowPath(source, sink)
select sink.getNode(), source, sink, "CWE-535: Exposure of Information Through Shell Error Message"