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
        not this instanceof CompileTimeConstantExpr and
        not v.getName().toLowerCase().matches("%encrypted%")
        )
    }
  }

  class SensitiveStringLiteral extends StringLiteral {
    SensitiveStringLiteral() {
      // Check for matches against the suspicious patterns
      exists(File f | 
        f = this.getCompilationUnit().getFile() and
        sensitiveStrings(f.getBaseName(), this.getValue())) and
      not exists(MethodCall mc |
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
         (mc.getMethod().hasName("addRequestProperty") and mc.getArgument(0) = this)
        )
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
        (sinks(f.getBaseName(), mc.getMethod().getName(), "I/O Sink") or
        sinks(f.getBaseName(), mc.getMethod().getName(), "Print Sink") or
        sinks(f.getBaseName(), mc.getMethod().getName(), "Network Sink") or
        // sinks(f.getBaseName(), mc.getMethod().getName(), "Log Sink") or
        sinks(f.getBaseName(), mc.getMethod().getName(), "Database Sink") or
        sinks(f.getBaseName(), mc.getMethod().getName(), "Email Sink") or
        sinks(f.getBaseName(), mc.getMethod().getName(), "IPC Sink") or
        sinks(f.getBaseName(), mc.getMethod().getName(), "Clipboard Sink") or
        sinks(f.getBaseName(), mc.getMethod().getName(), "GUI Display Sink") or
        sinks(f.getBaseName(), mc.getMethod().getName(), "RPC Sink") or
        sinks(f.getBaseName(), mc.getMethod().getName(), "Environment Variable Sink") or
        sinks(f.getBaseName(), mc.getMethod().getName(), "Command Execution Sink") or
        sinks(f.getBaseName(), mc.getMethod().getName(), "Configuration File Sink"))

        and
        sink.asExpr() = mc.getAnArgument()
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
  


