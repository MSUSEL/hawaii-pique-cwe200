/**
 * @name CWE-214: Invocation of Process Using Visible Sensitive Information
 * @description Detects when sensitive information is passed to a process execution command.
 * @kind path-problem
 * @problem.severity error
 * @precision high
 * @id java/invocation-of-process-using-visible-sensitive-information/214
 * @tags security
 *       external/cwe/cwe-200
 * @cwe CWE-214
 */


import java
import semmle.code.java.dataflow.FlowSources
import semmle.code.java.dataflow.TaintTracking
import SensitiveInfo.SensitiveInfo

module Flow = TaintTracking::Global<ProcessExecutionWithSensitiveInfoConfig>;
import Flow::PathGraph

module ProcessExecutionWithSensitiveInfoConfig implements DataFlow::ConfigSig {

  predicate isSource(DataFlow::Node source) {
    exists(SensitiveVariableExpr sve |
      source.asExpr() = sve and
      not sve.(VarAccess).getVariable().getName().toLowerCase().matches("%encrypt%"))
    
    or 
    exists(SensitiveStringLiteral ssl |source.asExpr() = ssl )
  }

  predicate isSink(DataFlow::Node sink) {
    // Checks if the sink is a method call to Runtime.exec
    exists(MethodCall execCall |
      execCall.getMethod().getDeclaringType().hasQualifiedName("java.lang", "Runtime") and
      execCall.getMethod().hasName("exec") and
      sink.asExpr() = execCall.getAnArgument()
    )
    or
    // Checks if the sink is a method call to ProcessBuilder.start
    exists(MethodCall envCall |
      envCall.getMethod().getDeclaringType().hasQualifiedName("java.lang", "ProcessBuilder") and
      envCall.getMethod().hasName("start") and
      sink.asExpr() = envCall.getQualifier()
    ) 
    or
    // Checks if the sink is a method call to ProcessBuilder
    exists(MethodCall envCall |
      envCall.getMethod().hasName("ProcessBuilder") and
      sink.asExpr() = envCall.getQualifier()
    )
    or
    // Checks if the sink is a method call to ProcessBuilder.environment.put
    exists(MethodCall putCall, MethodCall envCall |
      putCall.getMethod().hasName("put") and
      envCall = putCall.getQualifier().(MethodCall) and
      envCall.getMethod().hasName("environment") and
      envCall.getQualifier().getType().(RefType).hasQualifiedName("java.lang", "ProcessBuilder") and
      (sink.asExpr() = putCall.getAnArgument())
    )
  }

predicate isBarrier(DataFlow::Node node) {
  exists(MethodCall mc |
    // Check if the method name contains 'sanitize' or 'encrypt', case-insensitive
    (mc.getMethod().getName().toLowerCase().matches("%sanitize%") or
    mc.getMethod().getName().toLowerCase().matches("%encrypt%"))
    and
  // Consider both arguments and the return of sanitization/encryption methods as barriers
  (node.asExpr() = mc.getAnArgument() or node.asExpr() = mc)
  )
}

}

from Flow::PathNode source, Flow::PathNode sink
where Flow::flowPath(source, sink)
select sink.getNode(), source, sink, "Sensitive information passed to process execution."
