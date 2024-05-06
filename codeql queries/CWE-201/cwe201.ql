/**
 * @name CWE-201: Information exposure through transmitted data
 * @description Transmitting sensitive information to the user is a potential security risk.
 * @kind problem
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
 
private class GetMessageFlowSource extends DataFlow::Node {
  GetMessageFlowSource() {
    exists(Method method | this.asExpr().(MethodCall).getMethod() = method |
      method.hasName("getMessage") and
      method.hasNoParameters() and
      method.getDeclaringType().hasQualifiedName("java.lang", "Throwable")
    )
  }
}



class MailSendMethod extends DataFlow::Node {
  MailSendMethod() {
    exists(MethodCall mailCall | this.asExpr() = mailCall.getAnArgument() | 
      mailCall.getMethod().hasName("setText") and
      mailCall.getQualifier().getType().(RefType).hasQualifiedName("javax.mail", "Message")
    )
  }
}
 
 module ExposureInTransmittedDataConfig implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node source){
    source.asExpr() instanceof SensitiveVariableExpr or source instanceof GetMessageFlowSource
  }

  predicate isSink(DataFlow::Node sink) {
    sink instanceof InformationLeakSink or 
    sink instanceof MailSendMethod or 
    sink.asExpr() instanceof UrlConstructorCall 
 }
}

 

 module ExposureInTransmittedData =TaintTracking::Global<ExposureInTransmittedDataConfig>;
 
 from ExposureInTransmittedData::PathNode source, ExposureInTransmittedData::PathNode sink
 where ExposureInTransmittedData::flowPath(source, sink)
 select sink.getNode(), "Transmission of Sensitive information might be exposed here."
 