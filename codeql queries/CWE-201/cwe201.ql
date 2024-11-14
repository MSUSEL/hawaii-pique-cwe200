/**
 * @name CWE-201: Information exposure through transmitted data
 * @description Transmitting sensitive information to the user is a potential security risk.
 * @kind path-problem
 * @problem.severity error
 * @security-severity 4.3
 * @precision high
 * @id java/transmitted-data-exposure/201
 * @tags security
 *       external/cwe/cwe-201
 * @cwe CWE-201
 */

 import java
 import semmle.code.java.security.SensitiveActions
 import semmle.code.java.dataflow.FlowSources
 import semmle.code.java.dataflow.TaintTracking
 import semmle.code.java.dataflow.DataFlow
 import semmle.code.java.frameworks.Networking
 private import semmle.code.java.security.InformationLeak
 import SensitiveInfo.SensitiveInfo
 import Barrier.Barrier
 
 module ExposureInTransmittedData = TaintTracking::Global<ExposureInTransmittedDataConfig>;
//  module Flow = TaintTracking::Global<HttpServletExceptionSourceConfig>;

import ExposureInTransmittedData::PathGraph

 module ExposureInTransmittedDataConfig implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node source){source.asExpr() instanceof SensitiveVariableExpr}

  predicate isSink(DataFlow::Node sink) {
    sink instanceof InformationLeakSink or 
    // getSink(sink, "Email Sink") or

    exists(MethodCall mc | 
      sink.asExpr() = mc.getAnArgument() and 
      mc.getMethod().hasName("write") and
      mc.getEnclosingCallable().getDeclaringType().hasQualifiedName("javax.servlet.http", "HttpServletResponse"))

    or
      
    exists(MethodCall mc | 
      sink.asExpr() = mc.getAnArgument() and 
      mc.getMethod().hasName("write") and
      (mc.getEnclosingCallable().getDeclaringType().hasQualifiedName("java.io", "OutputStream") or
      mc.getEnclosingCallable().getDeclaringType().hasQualifiedName("java.net", "Socket")))
 }

  predicate isBarrier(DataFlow::Node node) {
    Barrier::isBarrier(node)
}

 from ExposureInTransmittedData::PathNode source, ExposureInTransmittedData::PathNode sink
 where ExposureInTransmittedData::flowPath(source, sink)
  select sink.getNode(), source, sink,
  "CWE-201: Transmissions of sensitive information"
 

  