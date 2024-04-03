export const SensitiveVariablesContents:string=`
import java

private string suspicious() {
  result =
    [
        ======
    ]
}

predicate suspicious(string fileName, string variableName) {
----------
}


class SensitiveVariable extends Variable {
  SensitiveVariable() {
    this.getName().matches(suspicious())
  }
}

class SensitiveVariableExpr extends Expr {
  SensitiveVariableExpr() {
    exists(Variable v | this = v.getAnAccess() |
    v.getName().matches(suspicious()) and
      not this instanceof CompileTimeConstantExpr
    )
  }
}

class SensitiveStringLiteral extends StringLiteral {
  SensitiveStringLiteral() {
    // Check for matches against the suspicious patterns
    this.getValue().regexpMatch(suspicious()) and
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