/**
 * @name Exposure of sensitive information through Spring Boot REST controllers
 * @description Detects when sensitive information from exceptions or other sensitive sources
 *              is exposed to clients via Spring Boot REST controller responses, which could lead to
 *              information disclosure vulnerabilities.
 * @kind path-problem
 * @problem.severity warning
 * @id java/springboot-sensitive-info-exposure-enhanced
 * @tags security
 *       external/cwe/cwe-550
 *       external/cwe/cwe-200
 * @cwe CWE-550
 */
import java
import semmle.code.java.dataflow.DataFlow
import semmle.code.java.dataflow.TaintTracking

class SpringBootSensitiveInfoExposureConfig extends TaintTracking::Configuration {
  SpringBootSensitiveInfoExposureConfig() { this = "SpringBootSensitiveInfoExposureConfig" }

  override predicate isSource(DataFlow::Node source) {
    exists(MethodAccess ma |
      // Include Throwable methods and system/environment properties as sources
      (
      (ma.getMethod().getDeclaringType().getASupertype*().hasQualifiedName("java.lang", "Throwable") and
       ma.getMethod().hasName(["getMessage", "getStackTrace", "getStackTraceAsString", "printStackTrace", "toString"])) or
      (ma.getMethod().getDeclaringType().hasQualifiedName("java.lang", "System") and
       ma.getMethod().hasName(["getenv", "getProperty"])) or
      // Include user input as a potential source
      (ma.getMethod().getDeclaringType().hasQualifiedName("javax.servlet", "ServletRequest") and
       ma.getMethod().hasName(["getParameter", "getAttribute"])))
     and source.asExpr() = ma)
  }

  override predicate isSink(DataFlow::Node sink) {
    exists(MethodAccess ma |
      // ResponseStatusException constructor as a sink
      ma.getMethod().hasQualifiedName("org.springframework.web.server", "ResponseStatusException", "<init>") and
      sink.asExpr() = ma.getAnArgument()
    ) or
    exists(ConstructorCall cc |
      // Constructor call of ResponseStatusException as a sink
      cc.getConstructedType().hasQualifiedName("org.springframework.web.server", "ResponseStatusException") and
      sink.asExpr() = cc.getAnArgument()
    ) or
    exists(MethodAccess respMa |
      // ResponseEntity body method as a sink
      respMa.getMethod().getDeclaringType().hasQualifiedName("org.springframework.http", "ResponseEntity") and
      respMa.getMethod().hasName("body") and
      sink.asExpr() = respMa.getAnArgument()
    ) or
    // Additional constraint for logging methods to be considered as sinks
    exists(MethodAccess logMa |
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

from SpringBootSensitiveInfoExposureConfig config, DataFlow::Node source, DataFlow::Node sink
where config.hasFlow(source, sink)
select sink, "Sensitive information may be exposed to clients."
