import java

// Define the extensible predicate
extensible predicate sensitiveVariables(string fileName, string variableName);

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
