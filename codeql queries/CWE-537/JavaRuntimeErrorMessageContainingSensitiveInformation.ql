/**
 * @name Exposure of sensitive information in runtime error messages
 * @description Logging or printing sensitive information or detailed error messages can lead to information disclosure.
 * @kind path-problem
 * @problem.severity warning
 * @id CWE-537
 * @tags security
 *       external/cwe/cwe-537
 * @cwe CWE-537
 */

 import java
 import semmle.code.java.dataflow.TaintTracking
 import semmle.code.java.dataflow.FlowSources
 import semmle.code.java.security.SensitiveVariables

 module Flow = TaintTracking::Global<RuntimeSensitiveInfoExposureConfig>;
 import Flow::PathGraph
 
 module RuntimeSensitiveInfoExposureConfig implements DataFlow::ConfigSig{
  
  predicate isSource(DataFlow::Node source) {
     exists(MethodCall mc |
       // Sources from exceptions
       mc.getMethod().getDeclaringType().getASupertype*().hasQualifiedName("java.lang", "Throwable") and
       (mc.getMethod().hasName(["getMessage", "getStackTrace", "getStackTraceAsString", "printStackTrace"])) and
       source.asExpr() = mc
     )
     or
     exists(SensitiveVariableExpr sve |
      source.asExpr() = sve
    )
   }
 
  predicate isSink(DataFlow::Node sink) {
    exists(MethodCall mc |
      // Target method accesses that are calls to println on PrintStream
      mc.getMethod().getDeclaringType().hasQualifiedName("java.io", "PrintStream") and
      mc.getMethod().hasName("println") and
      isWithinCatchBlock(mc) and
      sink.asExpr() = mc.getAnArgument()
    )
     or
     exists(MethodCall mc |
       // Sinks using PrintWriter
       mc.getMethod().getDeclaringType().getASupertype*().hasQualifiedName("java.lang", "Throwable") and
       mc.getMethod().hasName("println") and
       mc.getQualifier().getType().(RefType).hasQualifiedName("java.io", "PrintWriter") and
       isWithinCatchBlock(mc) and
       sink.asExpr() = mc.getAnArgument()
     )
     or
     exists(MethodCall mc |
      mc.getMethod().hasName("printStackTrace") and
      sink.asExpr() = mc // Directly mark the method call as the sink
      )
     or
     exists(MethodCall log |
       log.getMethod().getDeclaringType().hasQualifiedName("org.apache.logging.log4j", "Logger") and
       log.getMethod().hasName(["error", "warn", "info", "debug", "fatal"]) and
       isWithinCatchBlock(log) and
       sink.asExpr() = log.getAnArgument()
    )
     or
     exists(MethodCall log |
       log.getMethod().getDeclaringType().hasQualifiedName("org.slf4j", "Logger") and
       log.getMethod().hasName(["error", "warn", "info", "debug"]) and
       isWithinCatchBlock(log) and
       sink.asExpr() = log.getAnArgument()
     )
   }

  predicate isBarrier(DataFlow::Node node) {
    exists(MethodCall mc |
      // Use regex matching to check if the method name contains 'sanitize', case-insensitive
      (mc.getMethod().getName().toLowerCase().matches("%sanitize%") or
      mc.getMethod().getName().toLowerCase().matches("%encrypt%") 
      )
      and
      node.asExpr() = mc.getAnArgument()
    )
  }  
}

predicate isWithinCatchBlock(MethodCall mc) {
  exists(CatchClause cc |
    mc.getEnclosingStmt().getEnclosingStmt*() = cc.getBlock()
  )
} 
 
from Flow::PathNode source, Flow::PathNode sink
where Flow::flowPath(source, sink)
select sink.getNode(), source, sink, "CWE-537: Java runtime error message containing sensitive information"