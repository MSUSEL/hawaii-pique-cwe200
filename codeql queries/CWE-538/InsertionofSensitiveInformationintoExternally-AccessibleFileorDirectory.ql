/**
 * @name CWE-538: Insertion of Sensitive Information into Externally-Accessible File or Directory
 * @description Detects when sensitive information, such as passwords or personal data, is written to files that may be accessible externally, potentially leading to information exposure.
 * @kind path-problem
 * @problem.severity error
 * @id CWE-538
 * @tags security
 *       external/cwe/cwe-538
 * @cwe CWE-538
 *  
 */


import java
import semmle.code.java.dataflow.TaintTracking
import semmle.code.java.security.SensitiveVariables
import DataFlow::PathGraph

/** A configuration for tracking flows of sensitive information to file writes. */
class SensitiveInfoToFileConfig extends TaintTracking::Configuration {
  SensitiveInfoToFileConfig() { this = "SensitiveInfoToFileConfig" }

  override predicate isSource(DataFlow::Node source) {
    // // Refine source identification
    // exists(Variable v |
    //   // Check for variable names that directly indicate sensitive information
    //   v.getName().regexpMatch("(?i).*(password|user|creditCard|db).*") or
    //   // You can expand this to include more patterns
    //   source.asExpr() = v.getAnAccess()
    // )

    exists(SensitiveVariableExpr sve |  source.asExpr() = sve) or
    exists(SensitiveStringLiteral ssl |  source.asExpr() = ssl)
  }

  override predicate isSink(DataFlow::Node sink) {
    // Refine sink identification
    exists(MethodAccess ma |
      (ma.getMethod().getDeclaringType().hasQualifiedName("java.io", "FileWriter") and
       ma.getMethod().hasName("write")) or
      (ma.getMethod().getDeclaringType().hasQualifiedName("java.io", "BufferedWriter") and
       ma.getMethod().hasName("write")) or
      (ma.getMethod().getDeclaringType().hasQualifiedName("java.io", "FileOutputStream") and
       ma.getMethod().hasName("write")) or
       ma.getMethod().hasName("write") or
       ma.getMethod().hasName("store")
      |
      sink.asExpr() = ma.getArgument(0) // Ensure the sensitive data is being written
    )
  }
}

from SensitiveInfoToFileConfig config, DataFlow::PathNode source, DataFlow::PathNode sink
where config.hasFlowPath(source, sink)
select sink.getNode(), source, sink, "CWE-538: Sensitive information written to an externally-accessible file."
