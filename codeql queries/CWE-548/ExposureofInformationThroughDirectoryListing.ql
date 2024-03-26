/**
 * @name Exposure of Directory Listing Information Through HTTP Responses
 * @description Detects potential exposure of directory listing information through HTTP responses in servlets, which could lead to sensitive information disclosure.
 * @kind path-problem
 * @problem.severity warning
 * @id CWE-548
 * @tags security
 *      external/cwe/cwe-548
 * @cwe CWE-548
 */

import java
import semmle.code.java.dataflow.TaintTracking
import semmle.code.java.frameworks.Servlets
import DataFlow::PathGraph

/** 
 * A configuration for tracking directory listing information exposure 
 * through HTTP responses in servlets.
 */
class DirectoryListingExposureConfig extends TaintTracking::Configuration {
  DirectoryListingExposureConfig() { this = "DirectoryListingExposureConfig" }

  override predicate isSource(DataFlow::Node source) {
    exists(MethodAccess ma |
      ma.getMethod().getDeclaringType().hasQualifiedName("java.io", "File") and
      ma.getMethod().hasName("listFiles") and
      source.asExpr() = ma
    )
  }

  override predicate isSink(DataFlow::Node sink) {
    exists(MethodAccess ma |
      (ma.getMethod().getDeclaringType().hasQualifiedName("java.io", "PrintWriter") and
      ma.getMethod().hasName("println"))and 
      // Ensure the argument to println is considered for the data flow.
      sink.asExpr() = ma.getArgument(0)
    ) 
    or
    exists(MethodAccess ma |
      ma.getMethod().getDeclaringType().getAnAncestor().hasQualifiedName("org.apache.logging.log4j", "Logger") and
      ma.getMethod().getName().matches("info|debug|warn|error|logger") and
      sink.asExpr() = ma.getArgument(0)

    )
  }
}

from DirectoryListingExposureConfig config, DataFlow::PathNode source, DataFlow::PathNode sink
where config.hasFlowPath(source, sink)
select sink.getNode(), source, sink, "Potential CWE-548: Directory listing information might be exposed."
