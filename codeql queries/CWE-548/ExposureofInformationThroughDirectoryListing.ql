import java
import semmle.code.java.dataflow.DataFlow

class HttpServletSubclass extends RefType {
    HttpServletSubclass() {
        this.getASupertype*().hasQualifiedName("javax.servlet.http", "HttpServlet")
    }
}

class ServletMethod extends Method {
    ServletMethod() {
        this.getDeclaringType() instanceof HttpServletSubclass
    }
}

from ServletMethod servletMethod, MethodAccess listFilesAccess, DataFlow::Node source, DataFlow::Node sink, MethodAccess printlnAccess
where
    listFilesAccess.getMethod().hasName("listFiles") and
    listFilesAccess.getMethod().getDeclaringType().hasQualifiedName("java.io", "File") and
    listFilesAccess.getEnclosingCallable() = servletMethod and
    
    printlnAccess.getMethod().hasName("println") and
    printlnAccess.getEnclosingCallable() = servletMethod 
    
    // and
    // printlnAccess.getReceiverType().getQualifiedName().matches("java.io.PrintStream|java.io.PrintWriter") 

    // and 
    
    // source.asExpr() = listFilesAccess and
    // sink.asExpr().(MethodAccess).getMethod().hasName("println") and
    // DataFlow::localFlow(source, sink)
select printlnAccess, "This method potentially exposes directory contents via the response."



