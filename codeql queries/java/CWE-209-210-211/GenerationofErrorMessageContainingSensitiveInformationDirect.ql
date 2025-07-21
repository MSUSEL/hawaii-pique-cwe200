/**
 * @name CWE-209: Generation of Error Message Containing Sensitive Information
 * @description Identifies instances where sensitive information such as file paths, usernames, or passwords might be included in error messages, potentially leading to information exposure.
 * @kind path-problem
 * @problem.severity warning
 * @id java/error-message-sensitive-info-direct/209
 * @tags security
 *       external/cwe/cwe-209
 * @cwe CWE-209
 */


 import java
 import semmle.code.java.dataflow.TaintTracking
 import semmle.code.java.dataflow.FlowSources
 import CommonSinks.CommonSinks
 import SensitiveInfo.SensitiveInfo
 import Barrier.Barrier
 private import semmle.code.java.security.InformationLeak

 
 module Flow = TaintTracking::Global<GenerationOfErrorMessageWithSensInfoConfig>;
 
 import Flow::PathGraph
 
 module GenerationOfErrorMessageWithSensInfoConfig implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node source) {
    exists(SensitiveVariableExpr sve |
      source.asExpr() = sve
    )
  }
 
  
   predicate isSink(DataFlow::Node sink) {
     // Consider the case where the sink exposes sensitive info within a catch clause of type RuntimeException
     exists(MethodCall mc, CatchClause cc |
       cc.getACaughtType().getASupertype*().hasQualifiedName("java.lang", "Throwable") and
       not cc.getACaughtType().getASupertype*().hasQualifiedName("java.lang", "RuntimeException") and
       not cc.getACaughtType().getASupertype*().hasQualifiedName("javax.servlet", "ServletException") and
       mc.getEnclosingStmt().getEnclosingStmt*() = cc.getBlock() and
       (
         CommonSinks::isPrintSink(sink) or
         CommonSinks::isErrorSink(sink) or
         CommonSinks::isServletSink(sink) or
         sink instanceof InformationLeakSink
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
 not isTestFile(sink.getNode().getLocation().getFile())
 select sink.getNode(), source, sink,
   "CWE-209: Sensitive information included in error messages."