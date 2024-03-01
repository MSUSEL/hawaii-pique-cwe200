import java
import semmle.code.java.dataflow.TaintTracking

 // Configuration for tracking the flow of sensitive information from SQLExceptions to the console.
class SqlExceptionToConsoleConfig extends TaintTracking::Configuration {
  SqlExceptionToConsoleConfig() { this = "SqlExceptionToConsoleConfig" }

  // Sources are calls to methods on SQLException that retrieve error messages or codes.
  override predicate isSource(DataFlow::Node source) {
    exists(MethodAccess ma |
      (ma.getReceiverType().(RefType).hasQualifiedName("java.sql", "SQLException") or
      ma.getReceiverType().(RefType).getASupertype*().hasQualifiedName("java.sql", "SQLException")or 
      // Extend sources to include common SQL-related exceptions from popular libraries
      ma.getReceiverType().(RefType).hasQualifiedName("org.springframework.jdbc", "DataAccessException") or
      ma.getReceiverType().(RefType).hasQualifiedName("org.hibernate", "HibernateException")) and
      ma.getMethod().hasName(["getMessage", "getStackTrace", "getStackTraceAsString", "getSQLState", 
      "getErrorCode", "toString", "getenv", "getProperty", "getParameter", "getAttribute"]) and
      source.asExpr() = ma
    )
  }
  
  // Sinks are calls to System.out.println that take a string as an argument.
  override predicate isSink(DataFlow::Node sink) {
    exists(MethodAccess ma |
      
      (
        (ma.getMethod().hasName("println")) or 
        
        // Direct console printing
        (ma.getMethod().hasName("println") and
        ma.getQualifier().getType().(RefType).hasQualifiedName("java.io", "PrintStream")) or
        
        // Logging frameworks
        (ma.getQualifier().getType().(RefType).hasQualifiedName("org.apache.logging.log4j", "Logger") and
        ma.getMethod().hasName(["error", "warn", "info", "debug"])) or
        
        // Error responses in web applications
        (ma.getMethod().hasName("sendError") and
        ma.getQualifier().getType().(RefType).hasQualifiedName("javax.servlet.http", "HttpServletResponse")) 
      )
      and
      sink.asExpr() = ma.getAnArgument()
    )
  }
}

from SqlExceptionToConsoleConfig config, DataFlow::PathNode source, DataFlow::PathNode sink
where config.hasFlowPath(source, sink)
select source, sink, "Potential exposure of SQLException information to the console."
