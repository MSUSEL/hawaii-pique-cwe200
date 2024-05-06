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
        or
        (exists(MethodCall println |
            println.getMethod().hasName("println") and
            println.getQualifier().(VarAccess).getVariable().getType() instanceof RefType and
            ((RefType)println.getQualifier().(VarAccess).getVariable().getType()).hasQualifiedName("java.io", "PrintStream") and
            sink.asExpr() = println.getAnArgument()) 
          )
    }

    predicate isErrorSink(DataFlow::Node sink) {
        exists(MethodCall getMessage |
            getMessage.getMethod().hasName(["getStackTrace", "getStackTraceAsString", "printStackTrace"]) and
            getMessage.getMethod().getDeclaringType().getASupertype*().hasQualifiedName("java.lang", "Throwable") and
            sink.asExpr() = getMessage
        )

        or

        exists(MethodCall mc |
            // Ensure the method call is 'printStackTrace'
            mc.getMethod().hasName("printStackTrace") and
            // Ensure the method is called on an instance of Throwable or its subclasses
            mc.getQualifier().getType().(RefType).getASupertype*().hasQualifiedName("java.lang", "Throwable") and
            // The sink is the method call itself, not an argument of the method call
            sink.asExpr() = mc
          )

          or

          exists(MethodCall ma |
            ma.getMethod().hasName("printStackTrace") and
            // Optionally check for a Throwable type or catch block context
            ma.getAnArgument().getType() instanceof TypeThrowable and
            sink.asExpr() = ma.getAnArgument()
        )
    }

    predicate isIOSink(DataFlow::Node sink) {
            // Refine sink identification
        exists(MethodCall mc |
            (mc.getMethod().getDeclaringType().hasQualifiedName("java.io", "FileWriter") and
            mc.getMethod().hasName("write")) or
            (mc.getMethod().getDeclaringType().hasQualifiedName("java.io", "BufferedWriter") and
            mc.getMethod().hasName("write")) or
            (mc.getMethod().getDeclaringType().hasQualifiedName("java.io", "FileOutputStream") and
            mc.getMethod().hasName("write")) or
            
            mc.getMethod().hasName("write") or
            mc.getMethod().hasName("store")
            |
            sink.asExpr() = mc.getAnArgument() // Ensure the sensitive data is being written
        )
    }

    predicate isSpringSink(DataFlow::Node sink) {
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
          )
    }
}
