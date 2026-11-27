kind: fixed
area: extending
page: extending/plugin-developer-guide.md

A plugin's build fingerprint now reports plugin API 1.4 (`api=1.4`), the version the Engine provides, instead of 1.3. Plugins built against API 1.2 or 1.3 still load; a plugin built with this SDK is refused by an Engine that provides API 1.3, with the reason, instead of loading interfaces that Engine does not have.
