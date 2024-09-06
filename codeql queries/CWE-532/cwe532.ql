/**
 * @name CWE-532: Log-Senistive-Information
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

 import java
private import semmle.code.java.dataflow.ExternalFlow
import semmle.code.java.dataflow.TaintTracking
import semmle.code.java.security.SensitiveActions
import semmle.code.java.frameworks.android.Compose
private import semmle.code.java.security.Sanitizers
import SensitiveInfo.SensitiveInfo

module PermissionBarrier {
  /**
   * Predicate to detect common permission-related methods
   */
  predicate isPermissionCheck(MethodAccess ma) {
    ma.getMethod().getName().matches("%Permission%") or
    ma.getMethod().getName().matches("%authorize%") or
    ma.getMethod().getName().matches("%canAccess%") or
    ma.getMethod().getName().matches("%isAuthorized%")
  }

  /**
   * Predicate to check if a node acts as a permission barrier
   */
  predicate isPermissionBarrier(DataFlow::Node node) {
    exists(MethodCall ma |
      isPermissionCheck(ma) and
      node.asExpr() = ma.getQualifier()
    )
  }
}

private class TypeType extends RefType {
  pragma[nomagic]
  TypeType() {
    this.getSourceDeclaration().getASourceSupertype*().hasQualifiedName("java.lang.reflect", "Type")
  }
}


module SensitiveLoggerConfig implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node source) { source.asExpr() instanceof SensitiveVariableExpr }

  predicate isSink(DataFlow::Node sink) { sinkNode(sink, "log-injection") }


  predicate isBarrier(DataFlow::Node sanitizer) {
    sanitizer.asExpr() instanceof LiveLiteral or
    sanitizer instanceof SimpleTypeSanitizer or
    sanitizer.getType() instanceof TypeType or
    PermissionBarrier::isPermissionBarrier(sanitizer)
  }

  predicate isBarrierIn(DataFlow::Node node) { isSource(node) }
}

module SensitiveLoggerFlow = TaintTracking::Global<SensitiveLoggerConfig>;
import SensitiveLoggerFlow::PathGraph

 
 from SensitiveLoggerFlow::PathNode source, SensitiveLoggerFlow::PathNode sink
 where SensitiveLoggerFlow::flowPath(source, sink)  and
 not exists (MethodCall ma |
             PermissionBarrier::isPermissionCheck(ma) and
             ma.getEnclosingCallable() = sink.getNode().getEnclosingCallable()
            )
 select sink.getNode(), source, sink, "This $@ is written to a log file.", source.getNode(),
   "potentially sensitive information"

