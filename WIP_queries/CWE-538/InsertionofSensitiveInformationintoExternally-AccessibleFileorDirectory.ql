import java
import semmle.code.java.dataflow.TaintTracking
import semmle.code.java.security.SensitiveVariables
import DataFlow::PathGraph


// /** Defines a class for tracking the flow of sensitive information to file writes. */
// class SensitiveInfoToFileConfig extends TaintTracking::Configuration {
//   SensitiveInfoToFileConfig() { this = "SensitiveInfoToFileConfig" }

//   override predicate isSource(DataFlow::Node source) {
//     exists(SensitiveVariableExpr sve |
//         source.asExpr() = sve
//       )
//   }

//   override predicate isSink(DataFlow::Node sink) {
//     exists(MethodAccess ma |
//     //   ma.getMethod().getDeclaringType().hasQualifiedName("java.io", "FileOutputStream") and
//     //   ma.getMethod().hasName("write") or
//       ma.getMethod().getDeclaringType().hasQualifiedName("java.io", "FileWriter") and
//       ma.getMethod().hasName("write") 
//     //   or
//     //   ma.getMethod().getDeclaringType().hasQualifiedName("java.io", "PrintWriter") and
//     //   ma.getMethod().hasName("print") or
//     //   ma.getMethod().hasName("println")

      
//       |
//       sink.asExpr() = ma.getArgument(0)
//     )
//   }
// }

// from SensitiveInfoToFileConfig config, DataFlow::PathNode source, DataFlow::PathNode sink
// where config.hasFlowPath(source, sink)
// select sink.getNode(), source, sink, "Potential CWE-538: Sensitive information written to an externally-accessible file."


from MethodAccess ma
where ma.getMethod().getName() = "write"
    and ma.getMethod().getDeclaringType().hasQualifiedName("java.io", "FileWriter")
select ma, "FileWriter.write usage detected."

