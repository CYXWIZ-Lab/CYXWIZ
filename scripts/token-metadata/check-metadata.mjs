/**
 * Check CYXWIZ token metadata on-chain
 */

import { createUmi } from '@metaplex-foundation/umi-bundle-defaults';
import {
  findMetadataPda,
  fetchMetadata,
  mplTokenMetadata
} from '@metaplex-foundation/mpl-token-metadata';
import { publicKey } from '@metaplex-foundation/umi';

const TOKEN_MINT = 'Az2YZ1hmY5iQ6Gi9rjTPRpNMvcyeYVt1PqjyRSRoNNYi';
const RPC_URL = 'https://api.devnet.solana.com';

async function main() {
  console.log('Checking CYXWIZ token metadata...\n');

  const umi = createUmi(RPC_URL).use(mplTokenMetadata());
  const mintPubkey = publicKey(TOKEN_MINT);

  // Find metadata PDA
  const metadataPda = findMetadataPda(umi, { mint: mintPubkey });
  console.log('Token Mint:', TOKEN_MINT);
  console.log('Metadata PDA:', metadataPda[0].toString());

  try {
    const metadata = await fetchMetadata(umi, metadataPda);
    console.log('\n--- On-Chain Metadata ---');
    console.log('Name:', metadata.name);
    console.log('Symbol:', metadata.symbol);
    console.log('URI:', metadata.uri || '(empty)');
    console.log('Seller Fee Basis Points:', metadata.sellerFeeBasisPoints);
    console.log('Update Authority:', metadata.updateAuthority.toString());
    console.log('Is Mutable:', metadata.isMutable);
    console.log('Primary Sale Happened:', metadata.primarySaleHappened);
  } catch (error) {
    console.log('\nNo metadata found or error:', error.message);
  }
}

main().catch(console.error);
