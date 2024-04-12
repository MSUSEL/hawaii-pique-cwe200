import java

from MethodCall mc
where
  mc.getMethod().hasName("printStackTrace") and
  mc.getQualifier().getType().(RefType).getASupertype*().hasQualifiedName("java.lang", "Throwable") 

select mc, "Method call to printStackTrace on an instance of Throwable or its subclasses."