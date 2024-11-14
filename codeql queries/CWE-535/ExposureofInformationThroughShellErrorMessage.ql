/**
 * @name CWE-535: Exposure of information through shell error message
 * @description Exposing error messages from shell commands can lead to information disclosure.
 * @kind path-problem
 * @problem.severity warning
 * @id java/shell-error-exposure/535
 * @tags security
 *       external/cwe/cwe-535
 * @cwe CWE-535
 */
 
import java
import semmle.code.java.dataflow.TaintTracking
import CommonSinks.CommonSinks
import SensitiveInfo.SensitiveInfo
import Barrier.Barrier

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
    CommonSinks::isErrPrintSink(sink) or
    CommonSinks::isErrorSink(sink) or
    CommonSinks::isServletSink(sink) or
    // Use the LLM response to indentify sinks
    getSinkAny(sink)
  }

  predicate isBarrier(DataFlow::Node node) {
    Barrier::isBarrier(node)
  } 
}

from Flow::PathNode source, Flow::PathNode sink
where Flow::flowPath(source, sink)
select sink.getNode(), source, sink, "CWE-535: Exposure of Information Through Shell Error Message"