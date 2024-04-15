/**
 * @name CWE-538: Insertion of Sensitive Information into Externally-Accessible File or Directory
 * @description Detects when sensitive information, such as passwords or personal data, is written to files that may be accessible externally, potentially leading to information exposure.
 * @kind path-problem
 * @problem.severity error
 * @id java/sensitive-info-to-file/538
 * @tags security
 *       external/cwe/cwe-538
 * @cwe CWE-538
 *  
 */


import java
import semmle.code.java.dataflow.TaintTracking
import shared.SensitiveVariables

module Flow = TaintTracking::Global<SensitiveInfoToFileConfig>;
import Flow::PathGraph

/** A configuration for tracking flows of sensitive information to file writes. */
module SensitiveInfoToFileConfig implements DataFlow::ConfigSig{

  predicate isSource(DataFlow::Node source) {
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

  predicate isSink(DataFlow::Node sink) {
    // Refine sink identification
    exists(MethodCall mc |
      (mc.getMethod().getDeclaringType().hasQualifiedName("java.io", "FileWriter") and
       mc.getMethod().hasName("write")) or
      (mc.getMethod().getDeclaringType().hasQualifiedName("java.io", "BufferedWriter") and
       mc.getMethod().hasName("write")) or
      (mc.getMethod().getDeclaringType().hasQualifiedName("java.io", "FileOutputStream") and
       mc.getMethod().hasName("write")) or
       mc.getMethod().hasName("write") or
       mc.getMethod().hasName("store")
      |
      sink.asExpr() = mc.getArgument(0) // Ensure the sensitive data is being written
    )
  }
}

from Flow::PathNode source, Flow::PathNode sink
where Flow::flowPath(source, sink)
select sink.getNode(), source, sink, "CWE-538: Sensitive information written to an externally-accessible file."
