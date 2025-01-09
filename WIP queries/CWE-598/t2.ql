import java

from AddExpr concatOp, Expr param
where
  (param = concatOp.getLeftOperand() or param = concatOp.getRightOperand()) and
  // Check if the string contains either `?` or `&`, and also `=`
  (param.toString().indexOf("?") >= 0 or param.toString().indexOf("&") >= 0) and
  param.toString().indexOf("=") > 0
select param, param.toString(), "Query parameter detected."
