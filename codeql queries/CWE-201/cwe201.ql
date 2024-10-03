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
    exists(MethodCall mailCall | 
      this.asExpr() = mailCall.getAnArgument() and  
      (mailCall.getMethod().hasName("setText") or
      mailCall.getMethod().hasName("setContent") or
      mailCall.getMethod().hasName("setSubject") or 
      mailCall.getMethod().hasName("addRecipient") or
      mailCall.getMethod().hasName("setFrom") or
      mailCall.getMethod().hasName("addHeader"))
      
      and

      mailCall.getMethod().getDeclaringType().hasQualifiedName("javax.mail.internet", _) 
      
    )
  }
}
 
 module ExposureInTransmittedDataConfig implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node source){
    source.asExpr() instanceof SensitiveVariableExpr 
    // or source instanceof GetMessageFlowSource
  }

  predicate isSink(DataFlow::Node sink) {
    sink instanceof InformationLeakSink or 
    sink instanceof MailSendMethod or 
    getSink(sink, "Email Sink") or
    getSink(sink, "HTTP Sink") or


    exists(MethodCall mc | 
      sink.asExpr() = mc.getAnArgument() and 
      mc.getMethod().hasName("sendRedirect") or
      mc.getMethod().hasName("write") and
      mc.getEnclosingCallable().getDeclaringType().hasQualifiedName("javax.servlet.http", "HttpServletResponse"))

    or
      
    exists(MethodCall mc | 
      sink.asExpr() = mc.getAnArgument() and 
      mc.getMethod().hasName("write") and
      mc.getEnclosingCallable().getDeclaringType().hasQualifiedName("java.io", "OutputStream") or
      mc.getEnclosingCallable().getDeclaringType().hasQualifiedName("java.net", "Socket"))
 }

  predicate isBarrier(DataFlow::Node node) {
    exists(MethodCall mc |
      // Check if the method name contains 'sanitize' or 'encrypt', case-insensitive
      (mc.getMethod().getName().toLowerCase().matches("%sanitize%") or
      mc.getMethod().getName().toLowerCase().matches("%encrypt%")) and
    // Consider both arguments and the return of sanitization/encryption methods as barriers
    (node.asExpr() = mc.getAnArgument() or node.asExpr() = mc)
    )
  }
}

 

 module ExposureInTransmittedData =TaintTracking::Global<ExposureInTransmittedDataConfig>;
 
 from ExposureInTransmittedData::PathNode source, ExposureInTransmittedData::PathNode sink
 where ExposureInTransmittedData::flowPath(source, sink)
 select sink.getNode(), "Transmission of Sensitive information might be exposed here."
 