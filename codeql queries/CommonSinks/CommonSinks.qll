// This is a helper lib that contains a set of common sinks. This is not exhaustive and should be used with ChatGPT.

import java
import semmle.code.java.dataflow.DataFlow

module CommonSinks {

    predicate isLoggingSink(DataFlow::Node sink) {
        exists(MethodCall log |
          log.getMethod().getDeclaringType().hasQualifiedName("org.apache.logging.log4j", "Logger") and
          log.getMethod().hasName(["error", "warn", "info", "debug", "fatal"]) and
          sink.asExpr() = log.getAnArgument()
        )
        or
        exists(MethodCall log |
          log.getMethod().getDeclaringType().hasQualifiedName("org.slf4j", "Logger") and
          log.getMethod().hasName(["error", "warn", "info", "debug"]) and
          sink.asExpr() = log.getAnArgument()
        )
    }

    predicate isServletSink(DataFlow::Node sink) {
        exists(MethodCall mc |
            // Includes HttpServletResponse methods that can expose information
            (mc.getMethod().hasQualifiedName("javax.servlet.http", "HttpServletResponse", "sendError") or
            mc.getMethod().hasQualifiedName("javax.servlet.http", "HttpServletResponse", "addHeader") or
            mc.getMethod().hasQualifiedName("javax.servlet.http", "HttpServletResponse", "setStatus")) and
            sink.asExpr() = mc.getAnArgument()
        )
        or
        exists(MethodCall mc |
            // Targets servlet response writing methods
            mc.getMethod().hasName("write") and
            // Checks the write method is called on a PrintWriter obtained from HttpServletResponse
            mc.getQualifier().(MethodCall).getMethod().hasName("getWriter") and
            mc.getQualifier().(MethodCall).getQualifier().getType().(RefType).hasQualifiedName("javax.servlet.http", "HttpServletResponse") and
            sink.asExpr() = mc.getAnArgument()
        )
    }

    predicate isPrintSink(DataFlow::Node sink) {
        exists(MethodCall mc |
            // Targets PrintWriter methods that may leak information
            mc.getMethod().hasName("println") and
            mc.getQualifier().getType().(RefType).hasQualifiedName("java.io", "PrintWriter") and
            sink.asExpr() = mc.getAnArgument()
        )
        
    }

}
