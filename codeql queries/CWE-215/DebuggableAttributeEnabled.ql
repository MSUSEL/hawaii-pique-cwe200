/**
 * @name Android debuggable attribute enabled
 * @description An enabled debugger can allow for entry points in the application or reveal sensitive information.
 * @kind problem
 * @problem.severity warning
 * @security-severity 7.2
 * @id CWE-215
 * @tags security
 *       external/cwe/cwe-489
 * @precision very-high
 * @cwe CWE-215
 */

import java
import semmle.code.xml.AndroidManifest

from AndroidApplicationXmlElement androidAppElem
where
  androidAppElem.isDebuggable() and
  not androidAppElem.getFile().(AndroidManifestXmlFile).isInBuildDirectory()
select androidAppElem.getAttribute("debuggable"), "The 'android:debuggable' attribute is enabled."
