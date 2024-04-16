import java
import SensitiveInfo

class SensitiveVariableExpr extends Expr {
    SensitiveVariableExpr() {
      exists(Variable v, File f | this = v.getAnAccess() and
        f = v.getCompilationUnit().getFile() 
        and
        v.getName().matches(SensitiveInfo::thisIsATest(f.getBaseName())) and
        not this instanceof CompileTimeConstantExpr
      )
    }
  }


from SensitiveVariableExpr sve
select sve