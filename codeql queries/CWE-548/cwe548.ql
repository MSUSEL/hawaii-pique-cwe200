/**
 * @name Directories and files exposure
 * @description A directory listing provides an attacker with the complete
 *              index of all the resources located inside of the complete web
 *              directory, which could yield files containing sensitive
 *              information like source code and credentials to the attacker.
 * @kind problem
 * @problem.severity warning
 * @precision medium
 * @id java/server-directory-listing
 * @tags security
 *       experimental
 *       external/cwe/cwe-548
 */

import java
import semmle.code.xml.WebXML


private class DefaultTomcatServlet extends WebServletClass {
  DefaultTomcatServlet() {
    this.getTextValue() = "org.apache.catalina.servlets.DefaultServlet" //Default servlet of Tomcat and other servlet containers derived from Tomcat like Glassfish
  }
}


class DirectoryListingInitParam extends WebXmlElement {
  DirectoryListingInitParam() {
    this.getName() = "init-param" and
    this.getAChild("param-name").getTextValue() = "listings" and
    exists(WebServlet servlet |
      this.getParent() = servlet and
      servlet.getAChild("servlet-class") instanceof DefaultTomcatServlet
    )
  }

  predicate isListingEnabled() {
    this.getAChild("param-value").getTextValue().toLowerCase() = "true"
  }
}

from DirectoryListingInitParam initp
where initp.isListingEnabled()
select initp, "Directory listing should be disabled to mitigate filename and path disclosure."
