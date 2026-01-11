/**
 * Update CYXWIZ token metadata URI with image
 *
 * Usage: node update-metadata-uri.mjs <JSON_METADATA_URI>
 *
 * Steps to add token logo:
 * 1. Host cyxtoken.png on IPFS (use NFT.Storage, Pinata, or Arweave)
 * 2. Create a JSON file with token metadata:
 *    {
 *      "name": "CYXWIZ",
 *      "symbol": "CYXWIZ",
 *      "description": "CYXWIZ - Decentralized ML Compute Platform Token",
 *      "image": "ipfs://<YOUR_IMAGE_CID>"
 *    }
 * 3. Host the JSON file on IPFS
 * 4. Run: node update-metadata-uri.mjs ipfs://<YOUR_JSON_CID>
 */

import { createUmi } from '@metaplex-foundation/umi-bundle-defaults';
import {
  updateMetadataAccountV2,
  findMetadataPda,
  mplTokenMetadata
} from '@metaplex-foundation/mpl-token-metadata';
import {
  createSignerFromKeypair,
  signerIdentity,
  publicKey
} from '@metaplex-foundation/umi';
import fs from 'fs';

// Configuration
const KEYPAIR_PATH = 'C:/Users/chick/.config/solana/id.json';
const TOKEN_MINT = 'Az2YZ1hmY5iQ6Gi9rjTPRpNMvcyeYVt1PqjyRSRoNNYi';
const RPC_URL = 'https://api.devnet.solana.com';

// Token Metadata
const TOKEN_NAME = 'CYXWIZ';
const TOKEN_SYMBOL = 'CYXWIZ';

async function main() {
  const args = process.argv.slice(2);

  if (args.length === 0) {
    console.log('Usage: node update-metadata-uri.mjs <JSON_METADATA_URI>');
    console.log('');
    console.log('Example: node update-metadata-uri.mjs ipfs://QmXxx.../metadata.json');
    console.log('');
    console.log('Steps to add token logo:');
    console.log('1. Host cyxtoken.png on IPFS (use https://nft.storage)');
    console.log('2. Create metadata.json with image URL');
    console.log('3. Host metadata.json on IPFS');
    console.log('4. Run this script with the metadata URI');
    process.exit(1);
  }

  const metadataUri = args[0];

  console.log('='.repeat(60));
  console.log('CYXWIZ Token Metadata URI Updater');
  console.log('='.repeat(60));

  // Load keypair
  console.log('\n1. Loading keypair from:', KEYPAIR_PATH);
  const keypairData = JSON.parse(fs.readFileSync(KEYPAIR_PATH, 'utf-8'));
  const keypairBytes = Uint8Array.from(keypairData);

  // Create UMI instance
  console.log('2. Connecting to Solana Devnet...');
  const umi = createUmi(RPC_URL).use(mplTokenMetadata());

  // Create signer from keypair
  const keypair = umi.eddsa.createKeypairFromSecretKey(keypairBytes);
  const signer = createSignerFromKeypair(umi, keypair);
  umi.use(signerIdentity(signer));

  console.log('   Payer address:', signer.publicKey.toString());

  // Token mint public key
  const mintPubkey = publicKey(TOKEN_MINT);
  console.log('3. Token mint:', TOKEN_MINT);
  console.log('4. New metadata URI:', metadataUri);

  // Find metadata PDA
  const metadataPda = findMetadataPda(umi, { mint: mintPubkey });
  console.log('5. Metadata PDA:', metadataPda[0].toString());

  // Update metadata
  console.log('\n6. Updating on-chain metadata...');

  try {
    const tx = await updateMetadataAccountV2(umi, {
      metadata: metadataPda,
      updateAuthority: signer,
      data: {
        name: TOKEN_NAME,
        symbol: TOKEN_SYMBOL,
        uri: metadataUri,
        sellerFeeBasisPoints: 0,
        creators: null,
        collection: null,
        uses: null,
      },
      isMutable: true,
      primarySaleHappened: false,
    }).sendAndConfirm(umi);

    console.log('\n' + '='.repeat(60));
    console.log('SUCCESS! Token metadata URI updated.');
    console.log('='.repeat(60));
    console.log('\nTransaction signature:', Buffer.from(tx.signature).toString('base64'));
    console.log('\nMetadata URI:', metadataUri);
    console.log('\nView on Solana Explorer:');
    console.log(`https://explorer.solana.com/address/${TOKEN_MINT}?cluster=devnet`);

  } catch (error) {
    console.error('\nError updating metadata:', error.message);
    console.error('\nFull error:', error);
    process.exit(1);
  }
}

main().catch(console.error);
