/**
 * @name CWE-209: Information exposure through an error message
 * @description Logging or printing sensitive information or detailed error messages can lead to information disclosure.
 * @kind problem
 * @problem.severity warning
 * @id java/runtime-error-info-exposure-printStackTrace/200
 * @tags security
 *       external/cwe/cwe-209
 * @cwe CWE-209
 */


import java

from MethodCall mc
where
  mc.getMethod().hasName("printStackTrace")

select mc, "Method call to printStackTrace exposes sensitive information."