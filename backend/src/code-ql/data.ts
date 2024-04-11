export const SensitiveVariablesContents:string=`
import java

string suspicious(string fileName) {
----------
}

string suspiciousStrings(string fileName) {
++++++++++
}



string suspiciousComments() {
  result = 
  [
    **********
  ]
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
      this.getValue().regexpMatch(suspiciousStrings(f.getBaseName()))    
      ) and
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

class SensitiveComment extends StringLiteral {
  SensitiveComment() {
    // Check for matches against the suspicious patterns
    exists(File f | 
      f = this.getCompilationUnit().getFile() and
      this.getValue().regexpMatch(suspiciousComments(f.getBaseName()))    
    ) 
  }   
}
`