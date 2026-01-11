# CYXWIZ Token Metadata Scripts

Scripts for managing CYXWIZ token metadata on Solana Devnet.

## Token Info

| Property | Value |
|----------|-------|
| **Token Mint** | `Az2YZ1hmY5iQ6Gi9rjTPRpNMvcyeYVt1PqjyRSRoNNYi` |
| **Name** | CYXWIZ |
| **Symbol** | CYXWIZ |
| **Decimals** | 9 |
| **Network** | Devnet |

## Scripts

### 1. add-metadata.mjs

Creates initial on-chain metadata for the token (name, symbol).

```bash
node add-metadata.mjs
```

**Status**: Already executed. Token now shows "CYXWIZ" on Solana Explorer.

### 2. update-metadata-uri.mjs

Updates the token metadata with a URI pointing to a JSON file containing the logo.

```bash
node update-metadata-uri.mjs <JSON_METADATA_URI>
```

## Adding Token Logo

To add the CYXWIZ logo to the token:

### Step 1: Upload Image to IPFS

Use [NFT.Storage](https://nft.storage) (free):

1. Go to https://nft.storage
2. Create an account and get an API key
3. Upload `cyxtoken.png` from the repo root
4. Copy the CID (e.g., `bafybeiabc123...`)

### Step 2: Create Metadata JSON

Create a file `metadata.json`:

```json
{
  "name": "CYXWIZ",
  "symbol": "CYXWIZ",
  "description": "CYXWIZ - Decentralized ML Compute Platform Token",
  "image": "ipfs://bafybeiabc123..."
}
```

### Step 3: Upload Metadata JSON to IPFS

Upload `metadata.json` to NFT.Storage and get the CID.

### Step 4: Update On-Chain Metadata

```bash
cd scripts/token-metadata
node update-metadata-uri.mjs ipfs://bafybei<YOUR_JSON_CID>
```

## Verification

After updating, view the token on Solana Explorer:
https://explorer.solana.com/address/Az2YZ1hmY5iQ6Gi9rjTPRpNMvcyeYVt1PqjyRSRoNNYi?cluster=devnet

The logo should appear within a few minutes as explorers fetch the metadata.

## Dependencies

```bash
npm install
```

Required packages:
- @metaplex-foundation/mpl-token-metadata
- @metaplex-foundation/umi
- @metaplex-foundation/umi-bundle-defaults
- @solana/web3.js
- bs58
