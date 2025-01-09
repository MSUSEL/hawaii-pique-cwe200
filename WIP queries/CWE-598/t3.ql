import java


from DataFlow::Node sink
where exists(ConstructorCall urlConstructor, MethodCall openConnectionCall, MethodCall setRequestMethod |
  urlConstructor.getConstructedType().hasQualifiedName("java.net", "URL") and
  urlConstructor.getAnArgument() = sink.asExpr() and
  openConnectionCall.getMethod().hasName("openConnection") and
  openConnectionCall.getMethod().getDeclaringType().hasQualifiedName("java.net", "URL") and
  DataFlow::localExprFlow(urlConstructor, openConnectionCall.getQualifier()) and
  setRequestMethod.getMethod().hasName("setRequestMethod") and
  ((StringLiteral)setRequestMethod.getArgument(0)).getValue() = "GET" and
  DataFlow::localExprFlow(openConnectionCall, setRequestMethod.getQualifier())
)
select sink, "Detected GET request sink."
