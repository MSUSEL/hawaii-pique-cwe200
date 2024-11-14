import java
import semmle.code.java.dataflow.TaintTracking


// Define the extensible predicates
extensible predicate sensitiveVariables(string fileName, string variableName);
extensible predicate sensitiveStrings(string fileName, string variableName);
extensible predicate sensitiveComments(string fileName, string variableName);
extensible predicate sinks(string fileName, string sinkName, string sinkType); 
  
  class SensitiveVariableExpr extends Expr {
    SensitiveVariableExpr() {
      exists(Variable v, File f |
        this = v.getAnAccess() and
        f = v.getCompilationUnit().getFile() and
        sensitiveVariables(f.getBaseName(), v.getName()) and
        (
        v.getName().toLowerCase() != "message" and
        v.getName().toLowerCase() != "messages" and
        v.getName().toLowerCase() != "msg" and
        v.getName().toLowerCase() != "msgs" and
        v.getName().toLowerCase() != "text" and
        v.getName().toLowerCase() != "texts" and
        v.getName().toLowerCase() != "data" and
        v.getName().toLowerCase() != "body" and
        v.getName().toLowerCase() != "request" and
        v.getName().toLowerCase() != "req" and
        v.getName().toLowerCase() != "context" and
        v.getName().toLowerCase() != "contents" and
        v.getName().toLowerCase() != "id"
        ) and

        /* Exclude exceptions, if an exception is sensitive, then it will have a different source flow into it. 
        That source should be the sensitive source, not the exception. */
        not (
          this.getType() instanceof RefType and
          this.getType().(RefType).getASupertype+().hasQualifiedName("java.lang", "Throwable")
        )
      )
    }
  }
  

  class SensitiveStringLiteral extends StringLiteral {
    SensitiveStringLiteral() {
        // Check for matches against the suspicious patterns
        exists(File f | 
            f = this.getCompilationUnit().getFile() and
            sensitiveStrings(f.getBaseName(), this.getValue())) and
        not (
            // Exclude specific method calls
            exists(MethodCall mc |
                mc.getAnArgument() = this and
                (
                    mc.getMethod().hasName("getenv") or
                    mc.getMethod().hasName("getParameter") or
                    mc.getMethod().hasName("getProperty") or
                    mc.getMethod().hasName("getInitParameter") or
                    mc.getMethod().hasName("getHeader") or
                    mc.getMethod().hasName("getCookie") or
                    mc.getMethod().hasName("getAttribute") or
                    mc.getMethod().hasName("getAuthType") or
                    mc.getMethod().hasName("getRemoteUser") or
                    mc.getMethod().hasName("getResource") or
                    mc.getMethod().hasName("getResourceAsStream") or
                    (mc.getMethod().hasName("addRequestProperty") and mc.getAnArgument() = this)
                )
            ) or
            // Exclude common non-sensitive patterns
            this.getValue().regexpMatch(".*example.*") or
            this.getValue().regexpMatch(".*test.*") or
            this.getValue().regexpMatch(".*demo.*") or
            this.getValue().regexpMatch(".*foo.*") or 
            this.getValue().regexpMatch(".*bar.*") or
            this.getValue().regexpMatch(".*baz.*") or
            this.getValue().regexpMatch(".*secret.*") or
            // Exclude empty strings
            this.getValue() = "" or
            // Exclude whitespace-only strings
            this.getValue().regexpMatch("^\\s*$") or
            // Exclude strings with exactly one dot followed by a digit
            this.getValue().regexpMatch("^[^.]*\\.[0-9]+$") 
        ) and
        not exists(Annotation ann |
            ann = this.getParent() 
            // and
            // ann.getType().hasQualifiedName(_, "Issue")
        )
    }
}






  // class SensitiveComment extends StringLiteral {
  //   SensitiveComment() {
  //     exists(File f, string pattern |
  //       f = this.getCompilationUnit().getFile() and
  //       sensitiveComments(f.getBaseName(), pattern) and
  //       this.getValue().regexpMatch(pattern)
  //     )
  //   }   
  // }

  
    predicate getSink(DataFlow::Node sink, string sinkType) { 
      exists(File f, MethodCall mc | 
        f = mc.getFile() and
        sinks(f.getBaseName(), mc.getMethod().getName(), sinkType) and
        sink.asExpr() = mc.getAnArgument()
        )
    }

    predicate getSinkAny(DataFlow::Node sink) { 
      exists(File f, MethodCall mc | 
        f = mc.getFile() and
        (sinks(f.getBaseName(), mc.getMethod().getName(), _) and
        not sinks(f.getBaseName(), mc.getMethod().getName(), "Log Sink"))

        // (sinks(f.getBaseName(), mc.getMethod().getName(), "I/O Sink") or
        // sinks(f.getBaseName(), mc.getMethod().getName(), "Print Sink") or
        // sinks(f.getBaseName(), mc.getMethod().getName(), "Network Sink") or
        // // sinks(f.getBaseName(), mc.getMethod().getName(), "Log Sink") or
        // sinks(f.getBaseName(), mc.getMethod().getName(), "Database Sink") or
        // sinks(f.getBaseName(), mc.getMethod().getName(), "Email Sink") or
        // sinks(f.getBaseName(), mc.getMethod().getName(), "IPC Sink") or
        // sinks(f.getBaseName(), mc.getMethod().getName(), "Clipboard Sink") or
        // sinks(f.getBaseName(), mc.getMethod().getName(), "GUI Display Sink") or
        // sinks(f.getBaseName(), mc.getMethod().getName(), "RPC Sink") or
        // sinks(f.getBaseName(), mc.getMethod().getName(), "Environment Variable Sink") or
        // sinks(f.getBaseName(), mc.getMethod().getName(), "Command Execution Sink") or
        // sinks(f.getBaseName(), mc.getMethod().getName(), "Configuration File Sink"))

        and
        sink.asExpr() = mc.getAnArgument()
        )
    }

    class DetectedMethodCall extends MethodCall {
      DetectedMethodCall() {
        // Check for matches against the sinks
        exists(File f |
          // this.getEnclosingCallable().getFile() = f and
          sinks(f.getBaseName(), this.getMethod().getName(), _)
        )
      }
    }

    class DetectedMethod extends Method {
      DetectedMethod() {
        // Match methods based on their simple name and the base name of the file where they are called
        exists(MethodCall mc, File f |
          mc.getMethod() = this and
          mc.getLocation().getFile().getBaseName() = f.getLocation().getFile().getBaseName() and
          sinks(f.getBaseName(), this.getName(), _)
        )
      }
    }
    
    

    predicate getAllSinks(DataFlow::Node sink) { 
      exists(File f, MethodCall mc | 
        f = mc.getFile() and
        (sinks(f.getBaseName(), mc.getMethod().getName(), "I/O Sink") or
        sinks(f.getBaseName(), mc.getMethod().getName(), "Print Sink") or
        sinks(f.getBaseName(), mc.getMethod().getName(), "Network Sink") or
        sinks(f.getBaseName(), mc.getMethod().getName(), "Log Sink") or
        sinks(f.getBaseName(), mc.getMethod().getName(), "Database Sink") or
        sinks(f.getBaseName(), mc.getMethod().getName(), "Email Sink") or
        sinks(f.getBaseName(), mc.getMethod().getName(), "IPC Sink") or
        sinks(f.getBaseName(), mc.getMethod().getName(), "Clipboard Sink") or
        sinks(f.getBaseName(), mc.getMethod().getName(), "GUI Display Sink") or
        sinks(f.getBaseName(), mc.getMethod().getName(), "RPC Sink") or
        sinks(f.getBaseName(), mc.getMethod().getName(), "Environment Variable Sink") or
        sinks(f.getBaseName(), mc.getMethod().getName(), "Command Execution Sink") or
        sinks(f.getBaseName(), mc.getMethod().getName(), "Configuration File Sink"))
        )
    }

string getSinkName() { 
  exists(File f, MethodCall mc | 
    f = mc.getFile() and
    (sinks(f.getBaseName(), mc.getMethod().getName(), "I/O Sink") or
    sinks(f.getBaseName(), mc.getMethod().getName(), "Print Sink") or
    sinks(f.getBaseName(), mc.getMethod().getName(), "Network Sink") or
    sinks(f.getBaseName(), mc.getMethod().getName(), "Log Sink") or
    sinks(f.getBaseName(), mc.getMethod().getName(), "Database Sink") or
    sinks(f.getBaseName(), mc.getMethod().getName(), "Email Sink") or
    sinks(f.getBaseName(), mc.getMethod().getName(), "IPC Sink") or
    sinks(f.getBaseName(), mc.getMethod().getName(), "Clipboard Sink") or
    sinks(f.getBaseName(), mc.getMethod().getName(), "GUI Display Sink") or
    sinks(f.getBaseName(), mc.getMethod().getName(), "RPC Sink") or
    sinks(f.getBaseName(), mc.getMethod().getName(), "Environment Variable Sink") or
    sinks(f.getBaseName(), mc.getMethod().getName(), "Command Execution Sink") or
    sinks(f.getBaseName(), mc.getMethod().getName(), "Configuration File Sink"))
    // Bind the result to the method name
    and result = mc.getMethod().getName()
  )
}
  


