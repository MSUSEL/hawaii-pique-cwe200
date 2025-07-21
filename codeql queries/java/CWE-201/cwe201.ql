/**
 * @name CWE-201: Insertion of Sensitive Information Into Sent Data
 * @description Information that is sensitive is sent to an external entity.
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
 import ExposureInTransmittedData::PathGraph

 module ExposureInTransmittedDataConfig implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node source){source.asExpr() instanceof SensitiveVariableExpr}

  predicate isSink(DataFlow::Node sink) {
    sink instanceof InformationLeakSink or 
    // getSink(sink, "Email Sink") or
      
// 1) HttpServletResponse.setHeader(name, value)
exists(MethodCall mc |
  mc.getMethod().hasName("setHeader") and
  mc.getMethod().getDeclaringType().getASupertype*()
    .hasQualifiedName("javax.servlet.http", "HttpServletResponse") and
  sink.asExpr() = mc.getAnArgument()
)

or

// 2) MimeMessage.addHeader(name, value)
exists(MethodCall mc |
  mc.getMethod().hasName("addHeader") and
  mc.getMethod().getDeclaringType().getASupertype*()
    .hasQualifiedName("javax.mail.internet", "MimeMessage") and
  sink.asExpr() = mc.getAnArgument()
)

or

// 3) MimeMessage.setText(text)
exists(MethodCall mc |
  mc.getMethod().hasName("setText") and
  mc.getMethod().getDeclaringType().getASupertype*()
    .hasQualifiedName("javax.mail.internet", "MimeMessage") and
  sink.asExpr() = mc.getAnArgument()
)

or

// 4) Transport.send(message)
exists(MethodCall mc |
  mc.getMethod().hasName("send") and
  mc.getMethod().getDeclaringType().hasQualifiedName("javax.mail", "Transport") and
  sink.asExpr() = mc.getAnArgument()
)

or

// 5) Raw sockets only: Socket.getOutputStream().write(...)
exists(MethodCall mc |
  mc.getMethod().hasName("write") and
  sink.asExpr() = mc.getAnArgument() and
  mc.getMethod().getDeclaringType().getASupertype*()
    .hasQualifiedName("java.net", "Socket")
)

or

// 6) FileOutputStream.write(...)
exists(MethodCall mc |
  mc.getMethod().hasName("write") and
  sink.asExpr() = mc.getAnArgument() and
  mc.getMethod().getDeclaringType().hasQualifiedName("java.io", "FileOutputStream")
)

or

// 7) Servlet‐response writers only (PrintWriter from HttpServletResponse.getWriter())
exists(MethodCall mc |
  (mc.getMethod().hasName("write") or mc.getMethod().hasName("println")) and
  sink.asExpr() = mc.getAnArgument() and
  mc.getMethod().getDeclaringType().getASupertype*()
    .hasQualifiedName("java.io", "Writer") and
  exists(MethodCall alloc |
    alloc.getMethod().hasName("getWriter") and
    alloc.getMethod().getDeclaringType().getASupertype*()
      .hasQualifiedName("javax.servlet.http", "HttpServletResponse") and
    alloc.getQualifier() = mc.getQualifier()
  )
)

or

// 8) XMLStreamWriter.writeAttribute(name, value)
exists(MethodCall mc |
  mc.getMethod().hasName("writeAttribute") and
  mc.getMethod().getDeclaringType().getASupertype*()
    .hasQualifiedName("javax.xml.stream", "XMLStreamWriter") and
  sink.asExpr() = mc.getArgument(1)
)
 }

  predicate isBarrier(DataFlow::Node node) {
    Barrier::barrier(node)
  }
}


predicate isTestFile(File f) {
  // Convert path to lowercase for case-insensitive matching
  exists(string path | path = f.getAbsolutePath().toLowerCase() |
    // Check for common test-related directory or file name patterns
    path.regexpMatch(".*(test|tests|testing|test-suite|testcase|unittest|integration-test|spec).*")
  )
}

 from ExposureInTransmittedData::PathNode source, ExposureInTransmittedData::PathNode sink
 where ExposureInTransmittedData::flowPath(source, sink)
 and not isTestFile(sink.getNode().getLocation().getFile())

  select sink.getNode(), source, sink,
  "CWE-201: Insertion of Sensitive Information Into Sent Data"
 

  