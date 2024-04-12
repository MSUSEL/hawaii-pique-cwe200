/**
 * @name Exposure of Directory Listing Information Through HTTP Responses
 * @description Detects potential exposure of directory listing information through HTTP responses in servlets, which could lead to sensitive information disclosure.
 * @kind path-problem
 * @problem.severity warning
 * @id java/directory-info-exposure-http/CWE-548
 * @tags security
 *      external/cwe/cwe-548
 * @cwe CWE-548
 */

import java
import semmle.code.java.dataflow.TaintTracking
import semmle.code.java.frameworks.Servlets

module Flow = TaintTracking::Global<DirectoryListingExposureConfig>;
import Flow::PathGraph
/** 
 * A configuration for tracking directory listing information exposure 
 * through HTTP responses in servlets.
 */
module DirectoryListingExposureConfig implements DataFlow::ConfigSig {
  predicate isSource(DataFlow::Node source) {
    exists(MethodCall mc |
      mc.getMethod().getDeclaringType().hasQualifiedName("java.io", "File") and
      mc.getMethod().hasName("listFiles") and
      source.asExpr() = mc
    )
  }

  predicate isSink(DataFlow::Node sink) {
    exists(MethodCall mc |
      (mc.getMethod().getDeclaringType().hasQualifiedName("java.io", "PrintWriter") and
      mc.getMethod().hasName("println"))and 
      // Ensure the argument to println is considered for the data flow.
      sink.asExpr() = mc.getAnArgument()
    ) 
    or
    exists(MethodCall mc |
      mc.getMethod().getDeclaringType().hasQualifiedName("org.apache.logging.log4j", "Logger") and
      mc.getMethod().getName().matches(["info", "debug", "warn", "error", "logger"]) and
      sink.asExpr() = mc.getAnArgument()
    )
    or
    exists(MethodCall logMa |
      // Adds logging methods as sinks, considering them potential leak points
      logMa.getMethod().getDeclaringType().hasQualifiedName("org.slf4j", "Logger") and
      logMa.getMethod().hasName(["error", "warn", "info", "debug"]) and
      sink.asExpr() = logMa.getAnArgument())
  }
}

from Flow::PathNode source, Flow::PathNode sink
where Flow::flowPath(source, sink)
select sink.getNode(), source, sink, "CWE-548: Directory listing information might be exposed."
