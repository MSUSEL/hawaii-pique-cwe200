import java
import semmle.code.java.security.SensitiveVariables

from Javadoc comment, SensitiveComment sc
where comment.toString().matches("%" + sc.getValue() + "%")
select comment, "CWE-615: This comment contains sensitive information."
