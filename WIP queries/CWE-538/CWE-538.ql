/**
 * @name CWE-538: Insecure temporary file creation
 * @description The software creates temporary files in an external directory that is accessible to actors who are not explicitly authorized to access the information.
 * @kind path-problem
 * @problem.severity warning
 * @id java/temp-dir-info-disclosure/538
 * @tags security
 *       external/cwe/cwe-538
 * @cwe CWE-538
 */

 import java
 import semmle.code.java.dataflow.TaintTracking
 import semmle.code.java.security.TempDirUtils
 import Barrier.Barrier
 
 module Flow = TaintTracking::Global<TempDirInfoDisclosureConfig>;
 import Flow::PathGraph
 
 module TempDirInfoDisclosureConfig implements DataFlow::ConfigSig {
   
   // Define the source as sensitive variables such as passwords
   predicate isSource(DataFlow::Node source) {
     exists(SensitiveVariableExpr sve | source.asExpr() = sve)
   }
 
   // Define the sink as files written in the temporary directory
   predicate isSink(DataFlow::Node sink) {
     exists(MethodAccess ma, Expr tempDirExpr, File f |
    //    ma.getMethod().getName() = "write" and
       f.getName().matches(".*") and
       // Ensure the file's parent directory is derived from `java.io.tmpdir`
       tempDirExpr instanceof MethodCallExpr and
       tempDirExpr.(MethodCallExpr).getMethod().getName() = "getProperty" and
       tempDirExpr.(MethodCallExpr).getQualifier().toString() = "java.lang.System" and
       tempDirExpr.getAnArgument().getStringValue() = "java.io.tmpdir" and
       f.getParent().toString() = tempDirExpr.getStringValue() and
       sink.asExpr() = ma.getArgument(0)
     )
   }
 
   predicate isBarrier(DataFlow::Node node) {
     Barrier::barrier(node)
   } 
 }
 
 // Flow configuration
 from Flow::PathNode source, Flow::PathNode sink
 where Flow::flowPath(source, sink)
 select sink.getNode(), source, sink, "CWE-538: Insecure temporary file creation: sensitive information written to a file in an insecure temp directory."
 