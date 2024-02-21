import java
import semmle.code.java.dataflow.TaintTracking

/**
 * @name Exposure of information through shell error message
 * @description Exposing error messages from shell commands can lead to information disclosure.
 * @kind path-problem
 * @problem.severity warning
 * @id java/shell-error-exposure
 * @tags security
 *       external/cwe/cwe-535
 */
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
      and 
    not exists(MethodAccess sanitizeErrorOutput |
      sanitizeErrorOutput.getMethod().hasName("sanitizeErrorOutput") and
      sink.asExpr() = sanitizeErrorOutput.getAnArgument()
    ))
    or
    exists(MethodAccess getMessage |
      getMessage.getMethod().hasName(["getMessage", "getStackTrace", "getStackTraceAsString", "printStackTrace"]) and
      getMessage.getMethod().getDeclaringType().getASupertype*().hasQualifiedName("java.lang", "Throwable") and
      sink.asExpr() = getMessage
    )
  }
}

from ShellErrorExposureConfig config, DataFlow::PathNode source, DataFlow::PathNode sink
where config.hasFlowPath(source, sink)
select sink, source, sink, "Potential CWE-535: Exposure of information through shell error message"
