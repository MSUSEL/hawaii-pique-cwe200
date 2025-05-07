// Defines the structure of the sensitive variables data that is used in the .yml files.
export const SensitiveVariables:string=`
extensions:
  - addsTo:
      pack: custom-codeql-queries
      extensible: sensitiveVariables
    data:
    - ["fileName", "sensitiveVariable"]
----------
`