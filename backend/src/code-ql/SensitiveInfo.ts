export const SensitiveInfo:string=`
extensions:
  - addsTo:
      pack: custom-codeql-queries
      extensible: suspiciousVariables
    data:
    ----------

  - addsTo:
      pack: custom-codeql-queries
      extensible: suspiciousStrings
    data:
    ++++++++++

  - addsTo:
      pack: custom-codeql-queries
      extensible: suspiciousComments
    data:
    **********
`