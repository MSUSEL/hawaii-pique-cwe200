export const SensitiveVariablesContents:string=`
import java

private string suspicious() {
  result =
    [
        ======
    ]
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
`