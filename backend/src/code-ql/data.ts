export const SensitiveVariablesContents:string=`
import java

private string suspicious() {
  result =
    [
        ======
    ]
}


string suspicious(string fileName) {
----------
  }


class SensitiveVariable extends Variable {
  SensitiveVariable() {
    exists(File f | 
      f = this.getCompilationUnit().getFile() and
      this.getName().matches(suspicious(f.getBaseName()))
    )
  }
}

class SensitiveVariableExpr extends Expr {
  SensitiveVariableExpr() {
    exists(Variable v, File f | this = v.getAnAccess() and
      f = v.getCompilationUnit().getFile() and
      v.getName().matches(suspicious(f.getBaseName())) and
      not this instanceof CompileTimeConstantExpr
    )
  }
}

class SensitiveStringLiteral extends StringLiteral {
  SensitiveStringLiteral() {
    // Check for matches against the suspicious patterns
    exists(File f | 
      f = this.getCompilationUnit().getFile() and
      this.getValue().regexpMatch(suspicious(f.getBaseName()))    
      ) and
    not exists(MethodAccess ma |
      ma.getAnArgument() = this and
      (
        ma.getMethod().hasName("getenv") or
        ma.getMethod().hasName("getParameter") or
        ma.getMethod().hasName("getProperty") 
      )
    )
  }   
}
`