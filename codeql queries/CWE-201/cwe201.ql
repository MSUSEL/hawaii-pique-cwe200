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
      
    exists(MethodCall mc |
      sink.asExpr() = mc.getAnArgument() and
      mc.getMethod().hasName("write") and
      (mc.getEnclosingCallable().getDeclaringType().hasQualifiedName("java.io", "OutputStream") or
       mc.getEnclosingCallable().getDeclaringType().hasQualifiedName("java.io", "FileOutputStream") or
       mc.getEnclosingCallable().getDeclaringType().hasQualifiedName("java.net", "Socket") or
       mc.getEnclosingCallable().getDeclaringType().hasQualifiedName("javax.servlet.http", "HttpServletResponse")
      )
    ) or
  
    exists(MethodCall mc |
      sink.asExpr() = mc.getAnArgument() and
      (
       mc.getMethod().hasName("send") and mc.getEnclosingCallable().getDeclaringType().hasQualifiedName("javax.mail", "Transport")
      )
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
 

  