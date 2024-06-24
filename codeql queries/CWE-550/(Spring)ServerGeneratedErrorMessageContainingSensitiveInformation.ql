/**
 * @name CWE-550: Exposure of sensitive information through Spring Boot REST controllers
 * @description Detects when sensitive information from exceptions or other sensitive sources
 *              is exposed to clients via Spring Boot REST controller responses, which could lead to
 *              information disclosure vulnerabilities.
 * @kind path-problem
 * @problem.severity warning
 * @id java/spring-info-exposure/550
 * @tags security
 *       external/cwe/cwe-550
 *       external/cwe/cwe-200
 * @cwe CWE-550
 */
import java
import semmle.code.java.dataflow.DataFlow
import semmle.code.java.dataflow.TaintTracking
import CommonSinks.CommonSinks
import SensitiveInfo.SensitiveInfo

module Flow = TaintTracking::Global<SpringBootSensitiveInfoExposureConfig>;
import Flow::PathGraph

module SpringBootSensitiveInfoExposureConfig implements DataFlow::ConfigSig{

  predicate isSource(DataFlow::Node source) {
    exists(SensitiveVariableExpr sve | source.asExpr() = sve) or
     // Direct access to the exception variable itself
     exists(CatchClause cc | source.asExpr() = cc.getVariable().getAnAccess()) or
     // Consider any method call on the exception object as a source
     exists(CatchClause cc, MethodCall mc | mc.getQualifier() = cc.getVariable().getAnAccess() and source.asExpr() = mc)
   }
   
  predicate isSink(DataFlow::Node sink) {
    CommonSinks::isSpringSink(sink) or

    // Directly check for Spring annotations on the enclosing class of the log method
    exists(MethodCall logMa |
      sink.asExpr() = logMa.getAnArgument() and
      (
        logMa.getEnclosingCallable().getDeclaringType().getAnAnnotation().getType().hasQualifiedName("org.springframework.stereotype", "Controller") or
        logMa.getEnclosingCallable().getDeclaringType().getAnAnnotation().getType().hasQualifiedName("org.springframework.web.bind.annotation", "RestController") or
        logMa.getEnclosingCallable().getDeclaringType().getAnAnnotation().getType().hasQualifiedName("org.springframework.stereotype", "Service") or
        logMa.getEnclosingCallable().getDeclaringType().getAnAnnotation().getType().hasQualifiedName("org.springframework.stereotype", "Component")
      ) and 
      (
        CommonSinks::isPrintSink(sink) or
        CommonSinks::isErrorSink(sink) or
        CommonSinks::isIOSink(sink) or
        getSinkAny(sink) 
      )
    )
  }

  predicate isBarrier(DataFlow::Node node) {
    exists(MethodCall mc |
      // Check if the method name contains 'sanitize' or 'encrypt', case-insensitive
      (mc.getMethod().getName().toLowerCase().matches("%sanitize%") or
      mc.getMethod().getName().toLowerCase().matches("%encrypt%")) and
    // Consider both arguments and the return of sanitization/encryption methods as barriers
    (node.asExpr() = mc.getAnArgument() or node.asExpr() = mc)
    )
  }
}

from Flow::PathNode source, Flow::PathNode sink
where Flow::flowPath(source, sink)
select sink.getNode(), source, sink, "CWE-550: (Spring) Server Generated Error Message Containing Sensitive Information."
