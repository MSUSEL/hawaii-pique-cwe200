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

 import java
 import semmle.code.java.dataflow.TaintTracking
 import semmle.code.java.dataflow.FlowSources
 import semmle.code.java.security.SensitiveVariables

 
 class RuntimeSensitiveInfoExposureConfig extends TaintTracking::Configuration {
   RuntimeSensitiveInfoExposureConfig() { this = "RuntimeSensitiveInfoExposureConfig" }
 
   override predicate isSource(DataFlow::Node source) {
     exists(MethodAccess ma |
       // Sources from exceptions
       ma.getMethod().getDeclaringType().getASupertype*().hasQualifiedName("java.lang", "Throwable") and
       (ma.getMethod().hasName(["getMessage", "getStackTrace", "getStackTraceAsString", "printStackTrace"])) and
       source.asExpr() = ma
     )
     or
     exists(SensitiveVariableExpr sve |
      source.asExpr() = sve
    )
   }
 
   override predicate isSink(DataFlow::Node sink) {
    exists(MethodAccess ma |
      // Target method accesses that are calls to println on PrintStream
      ma.getMethod().getDeclaringType().hasQualifiedName("java.io", "PrintStream") and
      ma.getMethod().hasName("println") and
      isWithinCatchBlock(ma) and
      sink.asExpr() = ma.getAnArgument()
    )
     or
     exists(MethodAccess ma |
       // Sinks using PrintWriter
       ma.getMethod().getDeclaringType().getASupertype*().hasQualifiedName("java.lang", "Throwable") and
       ma.getMethod().hasName("println") and
       ma.getQualifier().getType().(RefType).hasQualifiedName("java.io", "PrintWriter") and
       isWithinCatchBlock(ma) and
       sink.asExpr() = ma.getAnArgument()
     )
     or
     exists(MethodAccess ma |
      ma.getMethod().hasName("printStackTrace") and
      sink.asExpr() = ma // Directly mark the method call as the sink
      )
     or
     exists(MethodAccess log |
       log.getMethod().getDeclaringType().hasQualifiedName("org.apache.logging.log4j", "Logger") and
       log.getMethod().hasName(["error", "warn", "info", "debug", "fatal"]) and
       isWithinCatchBlock(log) and
       sink.asExpr() = log.getAnArgument()
    )
     or
     exists(MethodAccess log |
       log.getMethod().getDeclaringType().hasQualifiedName("org.slf4j", "Logger") and
       log.getMethod().hasName(["error", "warn", "info", "debug"]) and
       isWithinCatchBlock(log) and
       sink.asExpr() = log.getAnArgument()
     )
   }

   override predicate isSanitizer(DataFlow::Node node) {
    exists(MethodAccess ma |
      // Use regex matching to check if the method name contains 'sanitize', case-insensitive
      (ma.getMethod().getName().toLowerCase().matches("%sanitize%") or
      ma.getMethod().getName().toLowerCase().matches("%encrypt%") 
      )
      and
      node.asExpr() = ma.getAnArgument()
    )
  }  

  /**
 * Checks if the given MethodAccess is within a CatchClause. This is important because we only want to consider error messages for this CWE.
 */
predicate isWithinCatchBlock(MethodAccess ma) {
  exists(CatchClause cc |
    ma.getEnclosingStmt().getEnclosingStmt*() = cc.getBlock()
  )
}

   
 }
 
 from RuntimeSensitiveInfoExposureConfig config, DataFlow::PathNode source, DataFlow::PathNode sink
 where config.hasFlowPath(source, sink)
 select sink, source, sink, "Potential CWE-537: Java runtime error message containing sensitive information"