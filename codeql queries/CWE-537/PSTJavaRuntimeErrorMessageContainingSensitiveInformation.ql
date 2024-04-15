/**
 * @name Exposure of sensitive information in runtime error messages
 * @description Logging or printing sensitive information or detailed error messages can lead to information disclosure.
 * @kind problem
 * @problem.severity warning
 * @id java/runtime-error-info-exposure-printStackTrace/537
 * @tags security
 *       external/cwe/cwe-537
 * @cwe CWE-537
 */


import java

from MethodCall mc
where
  mc.getMethod().hasName("printStackTrace") and
  mc.getQualifier().getType().(RefType).getASupertype*().hasQualifiedName("java.lang", "Throwable") 

select mc, "Method call to printStackTrace on an instance of Throwable or its subclasses."