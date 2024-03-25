import java
import semmle.code.java.dataflow.TaintTracking
import semmle.code.java.frameworks.Servlets

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
      sink.asExpr() = ma

    )
    // or
    // exists(MethodAccess ma |
    //   // Identify logging methods as sinks, focusing on debug logging
    //   ma.getMethod().getDeclaringType().getAnAncestor().hasQualifiedName("org.apache.logging.log4j", "Logger") and
    //   ma.getMethod().hasName("debug") and
    //   // Check for a specific pattern in the logged message
    //   ma.getArgument(0).(StringConcatenation).getAnOperand().(StringLiteral).getValue().matches("File size (bytes): %") and
    //   sink.asExpr() = ma.getArgument(0)
    // )
  }
}

from DirectoryListingExposureConfig config, DataFlow::PathNode source, DataFlow::PathNode sink
where config.hasFlowPath(source, sink)
select sink, source, sink, "Sensitive directory listing information flows to an exposure point."
