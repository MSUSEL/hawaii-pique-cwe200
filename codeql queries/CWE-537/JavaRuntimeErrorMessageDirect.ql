/**
 * @name CWE-537: Exposure of sensitive information in runtime error messages
 * @description Logging or printing sensitive information or detailed error messages can lead to information disclosure.
 * @kind path-problem
 * @problem.severity warning
 * @id java/runtime-error-info-exposure-direct/537
 * @tags security
 *       external/cwe/cwe-537
 * @cwe CWE-537
 */

 import java
 import semmle.code.java.dataflow.TaintTracking
 import semmle.code.java.dataflow.FlowSources
 import CommonSinks.CommonSinks
 import SensitiveInfo.SensitiveInfo
 import Barrier.Barrier
 
 module Flow = TaintTracking::Global<RuntimeSensitiveInfoExposureConfig>;
 
 import Flow::PathGraph
 
 module RuntimeSensitiveInfoExposureConfig implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node source) {
    exists(SensitiveVariableExpr sve |
      source.asExpr() = sve
    )
  }
 
  
   predicate isSink(DataFlow::Node sink) {
     // Consider the case where the sink exposes sensitive info within a catch clause of type RuntimeException
     exists(MethodCall mc, CatchClause cc |
       cc.getACaughtType().getASupertype*().hasQualifiedName("java.lang", "RuntimeException") and
       mc.getEnclosingStmt().getEnclosingStmt*() = cc.getBlock() and
       (
         CommonSinks::isPrintSink(sink) or
         CommonSinks::isErrorSink(sink) or
         CommonSinks::isServletSink(sink) 
        //  or
        //  getSinkAny(sink)
       ) and
       sink.asExpr() = mc.getAnArgument()
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
 
 from Flow::PathNode source, Flow::PathNode sink
 where Flow::flowPath(source, sink) and
 not isTestFile(sink.getNode().getLocation().getFile()) and
 select sink.getNode(), source, sink,
   "CWE-537: Java runtime error message containing sensitive information"