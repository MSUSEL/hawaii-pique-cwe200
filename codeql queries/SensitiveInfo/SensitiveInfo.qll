import java

// Define the extensible predicates
extensible predicate sensitiveVariables(string fileName, string variableName);
extensible predicate sensitiveStrings(string fileName, string variableName);
extensible predicate sensitiveComments(string variablesName);


module SensitiveVariables {
  class SensitiveVariableExpr extends Expr {
    SensitiveVariableExpr() {
      exists(Variable v, File f |
        this = v.getAnAccess() and
        f = v.getCompilationUnit().getFile() and
        sensitiveVariables(f.getBaseName(), v.getName()) and
        not this instanceof CompileTimeConstantExpr
      )
    }
  }
}

module SensitiveStrings{
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
          mc.getMethod().hasName("getProperty") 
        )
      )
    }   
  }
}

module SensitiveComments {
  class SensitiveComment extends StringLiteral {
    SensitiveComment() {
      exists(string pattern | 
        sensitiveComments(pattern) and 
        this.getValue().regexpMatch(pattern)
      )
    }   
  }
}

