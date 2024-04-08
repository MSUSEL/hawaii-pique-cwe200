import java
import semmle.code.java.security.SensitiveVariables

from Javadoc comment, SensitiveStringLiteral ssl
where comment.toString().matches("%" + ssl.getValue() + "%")
select comment, "CWE-615: This comment contains sensitive information."
