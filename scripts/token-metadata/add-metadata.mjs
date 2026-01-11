/**
 * Add metadata to CYXWIZ token on Solana Devnet
 *
 * Usage: node add-metadata.mjs
 *
 * Prerequisites:
 * - Solana keypair at C:/Users/chick/.config/solana/id.json
 * - Token mint authority must match the keypair
 */

import { createUmi } from '@metaplex-foundation/umi-bundle-defaults';
import {
  createMetadataAccountV3,
  mplTokenMetadata
} from '@metaplex-foundation/mpl-token-metadata';
import {
  createSignerFromKeypair,
  signerIdentity,
  publicKey
} from '@metaplex-foundation/umi';
import fs from 'fs';
import path from 'path';

// Configuration
const KEYPAIR_PATH = 'C:/Users/chick/.config/solana/id.json';
const TOKEN_MINT = 'Az2YZ1hmY5iQ6Gi9rjTPRpNMvcyeYVt1PqjyRSRoNNYi';
const RPC_URL = 'https://api.devnet.solana.com';

// Token Metadata
const TOKEN_NAME = 'CYXWIZ';
const TOKEN_SYMBOL = 'CYXWIZ';
const TOKEN_DESCRIPTION = 'CYXWIZ - Decentralized ML Compute Platform Token';
// TODO: Host cyxtoken.png on IPFS/Arweave and update this URL
const TOKEN_IMAGE = 'https://raw.githubusercontent.com/user/repo/main/cyxtoken.png';
const TOKEN_URI = ''; // Will be set to JSON metadata URI if we upload to IPFS

async function main() {
  console.log('='.repeat(60));
  console.log('CYXWIZ Token Metadata Creator');
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

  // Create metadata
  console.log('\n4. Creating on-chain metadata...');
  console.log('   Name:', TOKEN_NAME);
  console.log('   Symbol:', TOKEN_SYMBOL);

  try {
    const tx = await createMetadataAccountV3(umi, {
      mint: mintPubkey,
      mintAuthority: signer,
      payer: signer,
      updateAuthority: signer.publicKey,
      data: {
        name: TOKEN_NAME,
        symbol: TOKEN_SYMBOL,
        uri: TOKEN_URI,
        sellerFeeBasisPoints: 0,
        creators: null,
        collection: null,
        uses: null,
      },
      isMutable: true,
      collectionDetails: null,
    }).sendAndConfirm(umi);

    console.log('\n' + '='.repeat(60));
    console.log('SUCCESS! Token metadata created.');
    console.log('='.repeat(60));
    console.log('\nTransaction signature:', Buffer.from(tx.signature).toString('base64'));
    console.log('\nView on Solana Explorer:');
    console.log(`https://explorer.solana.com/address/${TOKEN_MINT}?cluster=devnet`);

  } catch (error) {
    if (error.message?.includes('already in use')) {
      console.log('\nMetadata account already exists. Updating instead...');
      // Metadata already exists - this is fine for devnet
      console.log('Token already has metadata. View on Explorer:');
      console.log(`https://explorer.solana.com/address/${TOKEN_MINT}?cluster=devnet`);
    } else {
      console.error('\nError creating metadata:', error.message);
      console.error('\nFull error:', error);
      process.exit(1);
    }
  }
}

main().catch(console.error);
