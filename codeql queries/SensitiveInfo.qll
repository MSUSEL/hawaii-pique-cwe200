import java


module SensitiveInfo {

// extensible string thisIsATest(string fileName);
extensible predicate thisIsATest(string fileName, string str);

class SensitiveVariableExpr extends Expr {
  SensitiveVariableExpr() {
    exists(Variable v, File f | this = v.getAnAccess() and
      f = v.getCompilationUnit().getFile() 
      and
      exists(v.getName().matches(thisIsATest(f.getBaseName(), v))) and
      not this instanceof CompileTimeConstantExpr
    )
  }
}

  // predicate isVariableSensitive(Variable v) {
  //   exists(MethodCall mc| )
  // }

}


