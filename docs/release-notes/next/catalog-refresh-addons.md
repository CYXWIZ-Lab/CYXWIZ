kind: fixed
area: extending
page: usage/addons.md

**Refresh in the CyxWiz Installer works with a catalog that lists an add-on.** The online refresh read each catalog entry as a compute pack or the CPU Engine only, so a signed catalog that listed an add-on was refused as a whole (*Downloaded manifest ... was not trusted*). It now reads add-on packages too. Once the signed catalog itself is verified, one package entry no longer stops the refresh: an entry whose details cannot be downloaded, do not match the signed catalog, or are in a format this version cannot read is listed as unavailable, and the rest of the catalog, with its blocked and revoked packages, is refreshed. The message says how many entries are not offered and why; refreshing again later restores them.
