/**
 * @name CWE-532: Log-Sensitive-Information
 * @description Writing sensitive information to log files can allow that
 *              information to be leaked to an attacker more easily.
 * @kind path-problem
 * @problem.severity warning
 * @security-severity 7.5
 * @precision medium
 * @id java/log-sensitive-information/532
 * @tags security
 *       external/cwe/cwe-532
 * @cwe CWE-532
 */

 private import semmle.code.java.dataflow.ExternalFlow
 import semmle.code.java.dataflow.TaintTracking
 import semmle.code.java.security.SensitiveActions
 import semmle.code.java.frameworks.android.Compose
 private import semmle.code.java.security.Sanitizers
 import SensitiveInfo.SensitiveInfo
  
 module SensitiveLoggerConfig implements DataFlow::ConfigSig {
   predicate isSource(DataFlow::Node source) {
    exists(Variable v |
      (
        v.getName() = "a" or
        v.getName() = "b" or
        v.getName() = "c" or
        v.getName() = "d" or
        v.getName() = "e" or
        v.getName() = "f" or
        v.getName() = "g"  
      )
        and
        source.asExpr() = v.getAnAccess()
      )   }
 
    predicate isSink(DataFlow::Node sink) { sinkNode(sink, "log-injection") }
}
 
 module SensitiveLoggerFlow = TaintTracking::Global<SensitiveLoggerConfig>;
 import SensitiveLoggerFlow::PathGraph
 
 from SensitiveLoggerFlow::PathNode source, SensitiveLoggerFlow::PathNode sink
 where SensitiveLoggerFlow::flowPath(source, sink)
 select sink.getNode(), source, sink, "This $@ is written to a log file.", source.getNode(),
    "potentially sensitive information"


