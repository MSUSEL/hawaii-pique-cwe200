import java
import semmle.code.java.dataflow.TaintTracking
import semmle.code.java.dataflow.FlowSources

/**
 * @name Exposure of sensitive information in runtime error messages
 * @description Logging or printing sensitive information or detailed error messages can lead to information disclosure.
 * @kind path-problem
 * @problem.severity warning
 * @id java/runtime-sensitive-info-exposure
 * @tags security
 *       external/cwe/cwe-537
 * @cwe CWE-537
 */
class RuntimeSensitiveInfoExposureConfig extends TaintTracking::Configuration {
  RuntimeSensitiveInfoExposureConfig() { this = "RuntimeSensitiveInfoExposureConfig" }

  override predicate isSource(DataFlow::Node source) {
    exists(MethodAccess ma |
      // Sources from exceptions
      ma.getMethod().getDeclaringType().getASupertype*().hasQualifiedName("java.lang", "Throwable") and
      (ma.getMethod().hasName(["getMessage", "getStackTrace", "getStackTraceAsString", "printStackTrace"])) and
      source.asExpr() = ma
    )
  }

  override predicate isSink(DataFlow::Node sink) {
    exists(MethodAccess ma |
      ma.getMethod().getDeclaringType().hasQualifiedName("java.io", "PrintStream") and
      ma.getMethod().hasName("println") and
      sink.asExpr() = ma.getAnArgument()
    )
    or
    exists(MethodAccess ma |
      // Sinks using PrintWriter
      ma.getMethod().hasName("println") and
      ma.getQualifier().getType().(RefType).hasQualifiedName("java.io", "PrintWriter") and
      sink.asExpr() = ma.getAnArgument()
    )
    or
    exists(MethodAccess ma |
      ma.getMethod().hasName("printStackTrace") and
      sink.asExpr() = ma
    )
    or
    exists(MethodAccess log |
      log.getMethod().getDeclaringType().hasQualifiedName("org.apache.logging.log4j", "Logger") and
      log.getMethod().hasName(["error", "warn", "info", "debug", "fatal"]) and
      sink.asExpr() = log.getAnArgument()
    )
    or
    exists(MethodAccess log |
      log.getMethod().getDeclaringType().hasQualifiedName("org.slf4j", "Logger") and
      log.getMethod().hasName(["error", "warn", "info", "debug"]) and
      sink.asExpr() = log.getAnArgument()
    )

  }
  
}

from RuntimeSensitiveInfoExposureConfig config, DataFlow::PathNode source, DataFlow::PathNode sink
where config.hasFlowPath(source, sink)
select sink, source, sink, "Potential CWE-537: Java runtime error message containing sensitive information"