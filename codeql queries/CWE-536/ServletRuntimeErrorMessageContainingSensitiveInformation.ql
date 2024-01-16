import java

// Find servlet methods that potentially expose sensitive error information
from Class servletClass, Method servletMethod, CatchClause cc, MethodAccess printlnOrLogCall
where
  servletClass.getASupertype*().hasQualifiedName("javax.servlet.http", "HttpServlet") and
  servletMethod.getDeclaringType() = servletClass and
  (servletMethod.hasName("doGet") or servletMethod.hasName("doPost")) and
  cc.getEnclosingCallable() = servletMethod and
  printlnOrLogCall.getEnclosingCallable() = cc.getEnclosingCallable() and
  (
    printlnOrLogCall.getMethod().hasName("println") or
    printlnOrLogCall.getMethod().hasName("getMessage") or
    printlnOrLogCall.getMethod().hasName("getStackTrace")
  )
select servletMethod.getBody(),
  "Potential CWE-536: Servlet Runtime Error Message Containing Sensitive Information"
