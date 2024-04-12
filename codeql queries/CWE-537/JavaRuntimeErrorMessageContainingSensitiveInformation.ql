/**
 * @name Exposure of sensitive information in runtime error messages
 * @description Logging or printing sensitive information or detailed error messages can lead to information disclosure.
 * @kind path-problem
 * @problem.severity warning
 * @id java/runtime-error-info-exposure/CWE-537
 * @tags security
 *       external/cwe/cwe-537
 * @cwe CWE-537
 */

 import java
 import semmle.code.java.dataflow.TaintTracking
 import semmle.code.java.dataflow.FlowSources
 import semmle.code.java.security.SensitiveVariables
 import CommonSinks.CommonSinks

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
    // Identify direct calls to printStackTrace as sinks.
    exists(MethodCall mc |
      mc.getMethod().hasName("printStackTrace") and
      mc.getQualifier().getType().(RefType).getASupertype*().hasQualifiedName("java.lang", "Throwable") and
      sink.asExpr() = mc
    )
    or
    // Identify logging, printing, or error handling operations as sinks using the CommonSinks module.
    exists(MethodCall mc, CatchClause cc | 
      mc.getEnclosingStmt().getEnclosingStmt*() = cc.getBlock() and
      (
        CommonSinks::isLoggingSink(sink) or
        CommonSinks::isPrintSink(sink) or
        CommonSinks::isErrorSink(sink)
      ) and
      sink.asExpr() = mc.getAnArgument()
    )
  }
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

 
from Flow::PathNode source, Flow::PathNode sink
where Flow::flowPath(source, sink)
select sink.getNode(), source, sink, "CWE-537: Java runtime error message containing sensitive information"
