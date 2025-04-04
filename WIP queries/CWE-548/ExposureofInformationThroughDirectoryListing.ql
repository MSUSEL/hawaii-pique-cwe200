/**
 * @name CWE-548: Exposure of Directory Listing Information Through HTTP Responses
 * @description Detects potential exposure of directory listing information through HTTP responses in servlets, excluding results that just expose the file name.
 * @kind path-problem
 * @problem.severity warning
 * @id java/directory-info-exposure-http/548
 * @tags security
 *      external/cwe/cwe-548
 * @cwe CWE-548
 */

 import java
 import semmle.code.java.dataflow.TaintTracking
 import semmle.code.java.frameworks.Servlets
 import CommonSinks.CommonSinks
 import Barrier.Barrier
 
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
     CommonSinks::isPrintSink(sink) or
     CommonSinks::isServletSink(sink)
   }
 
   predicate isBarrier(DataFlow::Node node) {
     Barrier::barrier(node)
   }
 }
 
 /**
  * Exclude flows where only the name of the file is exposed.
  */
 predicate isExcludedResult(Flow::PathNode source, Flow::PathNode sink) {
   // Check if the flow ends with getName and doesn't involve other sensitive information
   exists(MethodCall mc |
     source.getNode().asExpr() = mc.getQualifier() and
     mc.getMethod().hasName("getName") and
     sink.getNode().asExpr() = mc and
     // Ensure that the flow does not include sensitive directory paths
     not exists(MethodCall other |
       other.getQualifier() = source.getNode().asExpr() and
       other.getMethod().hasName("getAbsolutePath")
     )
   )
 }
 
 from Flow::PathNode source, Flow::PathNode sink
 where Flow::flowPath(source, sink) and not isExcludedResult(source, sink)
 select sink.getNode(), source, sink, "CWE-548: Directory listing information might be exposed."
 