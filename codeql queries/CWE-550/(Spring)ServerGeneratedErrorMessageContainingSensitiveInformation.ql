/**
 * @name Exposure of sensitive information through Spring Boot REST controllers
 * @description Detects when sensitive information from exceptions or other sensitive sources
 *              is exposed to clients via Spring Boot REST controller responses, which could lead to
 *              information disclosure vulnerabilities.
 * @kind path-problem
 * @problem.severity warning
 * @id java/spring-info-exposure/CWE-550
 * @tags security
 *       external/cwe/cwe-550
 *       external/cwe/cwe-200
 * @cwe CWE-550
 */
import java
import semmle.code.java.dataflow.DataFlow
import semmle.code.java.dataflow.TaintTracking
import CommonSinks.CommonSinks

module Flow = TaintTracking::Global<SpringBootSensitiveInfoExposureConfig>;
import Flow::PathGraph

module SpringBootSensitiveInfoExposureConfig implements DataFlow::ConfigSig{

  predicate isSource(DataFlow::Node source) {
    exists(MethodCall mc |
      // Include Throwable methods and system/environment properties as sources
      (
      (mc.getMethod().getDeclaringType().getASupertype*().hasQualifiedName("java.lang", "Throwable") and
      mc.getMethod().hasName(["getMessage", "getStackTrace", "getStackTraceAsString", "printStackTrace", "toString"])) or
      (mc.getMethod().getDeclaringType().hasQualifiedName("java.lang", "System") and
      mc.getMethod().hasName(["getenv", "getProperty"])) or
      // Include user input as a potential source
      (mc.getMethod().getDeclaringType().hasQualifiedName("javax.servlet", "ServletRequest") and
      mc.getMethod().hasName(["getParameter", "getAttribute"])))
     and source.asExpr() = mc)
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
        CommonSinks::isLoggingSink(sink) or
        CommonSinks::isPrintSink(sink) or
        CommonSinks::isErrorSink(sink) or
        CommonSinks::isIOSink(sink)
      )
    )
  }
}

from Flow::PathNode source, Flow::PathNode sink
where Flow::flowPath(source, sink)
select sink.getNode(), source, sink, "CWE-550: (Spring) Server Generated Error Message Containing Sensitive Information."
