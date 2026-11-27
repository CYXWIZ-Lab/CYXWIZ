kind: fixed
area: extending
page: usage/addons.md

**Refresh in the CyxWiz Installer works with a catalog that lists an add-on.** The online refresh read each catalog entry as a compute pack or the CPU Engine only, so a signed catalog that listed an add-on was refused as a whole (*Downloaded manifest ... was not trusted*). It now reads add-on packages too. An entry this version cannot read, such as a package in a newer format, is listed as unavailable and the rest of the catalog is refreshed; a downloaded file that is not the one the signed catalog names still stops the refresh and keeps the catalog you had.
