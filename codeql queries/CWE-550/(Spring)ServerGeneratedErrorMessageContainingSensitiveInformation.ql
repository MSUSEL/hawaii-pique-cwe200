/**
 * @name Exposure of sensitive information through Spring Boot REST controllers
 * @description Detects when sensitive information from exceptions or other sensitive sources
 *              is exposed to clients via Spring Boot REST controller responses, which could lead to
 *              information disclosure vulnerabilities.
 * @kind path-problem
 * @problem.severity warning
 * @id CWE-550
 * @tags security
 *       external/cwe/cwe-550
 *       external/cwe/cwe-200
 * @cwe CWE-550
 */
import java
import semmle.code.java.dataflow.DataFlow
import semmle.code.java.dataflow.TaintTracking
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
    exists(MethodCall mc |
      // ResponseStatusException constructor as a sink
      mc.getMethod().hasQualifiedName("org.springframework.web.server", "ResponseStatusException", "<init>") and
      sink.asExpr() = mc.getAnArgument()
    ) or
    exists(ConstructorCall cc |
      // Constructor call of ResponseStatusException as a sink
      cc.getConstructedType().hasQualifiedName("org.springframework.web.server", "ResponseStatusException") and
      sink.asExpr() = cc.getAnArgument()
    ) or
    exists(MethodCall respMa |
      // ResponseEntity body method as a sink
      respMa.getMethod().getDeclaringType().hasQualifiedName("org.springframework.http", "ResponseEntity") and
      respMa.getMethod().hasName("body") and
      sink.asExpr() = respMa.getAnArgument()
    ) or
    // Additional constraint for logging methods to be considered as sinks
    exists(MethodCall logMa |
        logMa.getMethod().getDeclaringType().hasQualifiedName("org.slf4j", "Logger") and
        logMa.getMethod().hasName(["error", "warn", "info", "debug"]) and
        sink.asExpr() = logMa.getAnArgument() and
        // Directly check for Spring annotations on the enclosing class of the log method
        (
          logMa.getEnclosingCallable().getDeclaringType().getAnAnnotation().getType().hasQualifiedName("org.springframework.stereotype", "Controller") or
          logMa.getEnclosingCallable().getDeclaringType().getAnAnnotation().getType().hasQualifiedName("org.springframework.web.bind.annotation", "RestController") or
          logMa.getEnclosingCallable().getDeclaringType().getAnAnnotation().getType().hasQualifiedName("org.springframework.stereotype", "Service") or
          logMa.getEnclosingCallable().getDeclaringType().getAnAnnotation().getType().hasQualifiedName("org.springframework.stereotype", "Component")
        )
      )
  }
}

from Flow::PathNode source, Flow::PathNode sink
where Flow::flowPath(source, sink)
select sink.getNode(), source, sink, "CWE-550: (Spring) Server Generated Error Message Containing Sensitive Information."
