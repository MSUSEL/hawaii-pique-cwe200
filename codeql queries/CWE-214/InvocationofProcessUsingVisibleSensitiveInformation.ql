/**
 * @name Invocation of Process Using Visible Sensitive Information
 * @description Detects when sensitive information is passed to a process execution command.
 * @kind path-problem
 * @problem.severity error
 * @precision high
 * @id java/invocation-of-process-using-visible-sensitive-information/CWE-214
 * @tags security
 *       external/cwe/cwe-200
 * @cwe CWE-214
 */


import java
import semmle.code.java.dataflow.FlowSources
import semmle.code.java.dataflow.TaintTracking
import semmle.code.java.security.SensitiveVariables

module Flow = TaintTracking::Global<ProcessExecutionWithSensitiveInfoConfig>;
import Flow::PathGraph

module ProcessExecutionWithSensitiveInfoConfig implements DataFlow::ConfigSig {

  predicate isSource(DataFlow::Node source) {
    exists(SensitiveVariableExpr sve | source.asExpr() = sve) or 
    exists(SensitiveStringLiteral ssl |source.asExpr() = ssl )
  }

  predicate isSink(DataFlow::Node sink) {
    // Checks if the sink is a method call to Runtime.exec
    exists(MethodCall execCall |
      execCall.getMethod().getDeclaringType().hasQualifiedName("java.lang", "Runtime") and
      execCall.getMethod().hasName("exec") and
      sink.asExpr() = execCall.getArgument(0)
    )
    or
    // Checks if the sink is a method call to ProcessBuilder.start
    exists(MethodCall envCall |
      envCall.getMethod().getDeclaringType().hasQualifiedName("java.lang", "ProcessBuilder") and
      envCall.getMethod().hasName("start") and
      sink.asExpr() = envCall.getQualifier()
    ) 
    or
    exists(MethodCall envCall |
      envCall.getMethod().hasName("ProcessBuilder") and
      sink.asExpr() = envCall.getQualifier()
    )
    or
    exists(MethodCall putCall, MethodCall envCall |
      putCall.getMethod().getDeclaringType().hasQualifiedName("java.util", "Map") and
      putCall.getMethod().hasName("put") and
      putCall.getQualifier() = envCall and
      envCall.getMethod().hasQualifiedName("java.lang", "ProcessBuilder", "environment") and
      sink.asExpr() = putCall.getArgument(0)
    )
  }
}

from Flow::PathNode source, Flow::PathNode sink
where Flow::flowPath(source, sink)
select sink.getNode(), source, sink, "Sensitive information passed to process execution."