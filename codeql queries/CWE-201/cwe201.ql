/**
 * @name Information exposure through transmitted data
 * @description Transmitting sensitive information to the user is a potential security risk.
 * @kind problem
 * @problem.severity error
 * @security-severity 4.3
 * @precision high
 * @id java/sensitive-data-transmission
 * @tags security
 *       external/cwe/cwe-201
 */

 import java
 import semmle.code.java.security.SensitiveActions
 import semmle.code.java.dataflow.FlowSources
 import semmle.code.java.dataflow.TaintTracking
 import semmle.code.java.dataflow.DataFlow
 private import semmle.code.java.security.InformationLeak
 import semmle.code.java.security.SensitiveVariables

private class GetMessageFlowSource extends DataFlow::Node {
  GetMessageFlowSource() {
    exists(Method method | this.asExpr().(MethodAccess).getMethod() = method |
      method.hasName("getMessage") and
      method.hasNoParameters() and
      method.getDeclaringType().hasQualifiedName("java.lang", "Throwable")
    )
  }
}
 
 module ExposureInTransmittedDataConfig implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node source){
    source.asExpr() instanceof SensitiveVariableExpr or source instanceof GetMessageFlowSource
  }

  predicate isSink(DataFlow::Node sink) { sink instanceof InformationLeakSink }
}

 

 module ExposureInTransmittedData =TaintTracking::Global<ExposureInTransmittedDataConfig>;
 
 from ExposureInTransmittedData::PathNode source, ExposureInTransmittedData::PathNode sink
 where ExposureInTransmittedData::flowPath(source, sink)
 select sink.getNode(), "Sensitive information might be exposed here."
 