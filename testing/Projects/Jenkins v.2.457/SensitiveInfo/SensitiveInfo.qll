import java

// Define the extensible predicates
extensible predicate sensitiveVariables(string fileName, string variableName);
extensible predicate sensitiveStrings(string fileName, string variableName);
extensible predicate sensitiveComments(string variablesName);


  class SensitiveVariableExpr extends Expr {
    SensitiveVariableExpr() {
      exists(Variable v, File f |
        this = v.getAnAccess() and
        f = v.getCompilationUnit().getFile() and
        sensitiveVariables(f.getBaseName(), v.getName()) and
        not this instanceof CompileTimeConstantExpr and
        not v.getName().toLowerCase().matches("%encrypt%")
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


  class SensitiveComment extends StringLiteral {
    SensitiveComment() {
      exists(string pattern | 
        sensitiveComments(pattern) and 
        this.getValue().regexpMatch(pattern)
      )
    }   
  }

