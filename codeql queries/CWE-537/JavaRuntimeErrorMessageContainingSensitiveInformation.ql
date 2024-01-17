import java

from CatchClause cc, MethodAccess ma, MethodAccess printlnCall, ReturnStmt returnStmt, 
     Expr printlnArg, AddExpr concatExpr, MethodAccess exceptionExposureCall, Stmt stmt,  Expr expr
     
where
    ma.getEnclosingCallable() = cc.getEnclosingCallable() and
    
    // Check if the sensitive informaiton is exposed via a print statement
    (
      printlnCall.getMethod().hasName("println") and
      printlnCall.getEnclosingCallable() = cc.getEnclosingCallable() and
      printlnCall.getEnclosingStmt() = cc.getBlock().getAChild*() and
      printlnArg = printlnCall.getAnArgument() and
      (
          // Direct call to Exposure
          (
              exceptionExposureCall = printlnArg and
              exceptionExposureCall instanceof MethodAccess and
              exceptionExposureCall.getMethod().hasName(["getMessage", "getStackTrace", "getStackTraceAsString", "printStackTrace"]) and
              exceptionExposureCall.getQualifier() instanceof VarAccess and
              exceptionExposureCall.getQualifier().(VarAccess).getVariable().getType() instanceof RefType and
              exceptionExposureCall.getQualifier().(VarAccess).getVariable().getType().(RefType).hasQualifiedName("java.lang", "Exception")
          ) or
          // Concatenation involving Exposure
          (
              printlnArg instanceof AddExpr and
              concatExpr = printlnArg and
              exceptionExposureCall = concatExpr.getAnOperand() and
              exceptionExposureCall instanceof MethodAccess and
              exceptionExposureCall.getMethod().hasName(["getMessage", "getStackTrace", "getStackTraceAsString", "printStackTrace"]) and
              exceptionExposureCall.getQualifier() instanceof VarAccess and
              exceptionExposureCall.getQualifier().(VarAccess).getVariable().getType() instanceof RefType and
              exceptionExposureCall.getQualifier().(VarAccess).getVariable().getType().(RefType).hasQualifiedName("java.lang", "Exception")
          )
      )
    )
    
    /* Still need to determine if return statements should be checked,
      and if so, should it be done in a sperate query or in this one?
    */

  //   or
  // Check if the sensitive informaiton is exposed via a return statement
  //   (
  //     // Iterate through statements in the catch block
  //     stmt = cc.getBlock().getAChild*() and
  //     stmt instanceof ReturnStmt and
  //     returnStmt = stmt and
  //     // Now iterate through the return statement's children
  //     stmt = returnStmt.getAChild*().(Stmt) and
  //     expr instanceof MethodAccess and
  //     // Ensure that expression is in the same method as the exec call
  //     expr.getEnclosingCallable() = ma.getEnclosingCallable() and
  //     // Find calls to an Exception that exposes information
  //     (
  //       expr.(MethodAccess).getMethod().hasName(["getMessage", "getStackTrace", "getStackTraceAsString", "printStackTrace"])
  //     ) and
  //     expr.(MethodAccess).getQualifier().(VarAccess).getVariable().getType() instanceof RefType and
  //     expr.(MethodAccess).getQualifier().(VarAccess).getVariable().getType().(RefType).hasQualifiedName("java.lang", "Exception")
  // )

select exceptionExposureCall, "Potential CWE-537: Java runtime error message containing sensitive information"
  