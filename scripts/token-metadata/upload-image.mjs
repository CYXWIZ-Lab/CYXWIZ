/**
 * Upload CYXWIZ token image to NFT.Storage (IPFS)
 *
 * Usage:
 *   1. Get a free API key from https://nft.storage
 *   2. Set NFT_STORAGE_KEY environment variable or pass as argument
 *   3. Run: node upload-image.mjs [API_KEY]
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const IMAGE_PATH = path.resolve(__dirname, '../../cyxtoken.png');

async function uploadToNFTStorage(apiKey) {
  console.log('='.repeat(60));
  console.log('CYXWIZ Token Image Upload to IPFS');
  console.log('='.repeat(60));

  // Check if image exists
  if (!fs.existsSync(IMAGE_PATH)) {
    console.error('Error: cyxtoken.png not found at:', IMAGE_PATH);
    process.exit(1);
  }

  console.log('\n1. Loading image from:', IMAGE_PATH);
  const imageData = fs.readFileSync(IMAGE_PATH);
  console.log('   Image size:', (imageData.length / 1024).toFixed(2), 'KB');

  console.log('\n2. Uploading to NFT.Storage (IPFS)...');

  const response = await fetch('https://api.nft.storage/upload', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'image/png',
    },
    body: imageData,
  });

  if (!response.ok) {
    const error = await response.text();
    console.error('Upload failed:', response.status, error);
    process.exit(1);
  }

  const result = await response.json();
  const cid = result.value.cid;

  console.log('\n' + '='.repeat(60));
  console.log('SUCCESS! Image uploaded to IPFS');
  console.log('='.repeat(60));
  console.log('\nIPFS CID:', cid);
  console.log('\nImage URLs:');
  console.log('  IPFS:', `ipfs://${cid}`);
  console.log('  Gateway:', `https://${cid}.ipfs.nftstorage.link`);
  console.log('  Alternative:', `https://nftstorage.link/ipfs/${cid}`);

  // Update metadata.json
  const metadataPath = path.join(__dirname, 'metadata.json');
  if (fs.existsSync(metadataPath)) {
    const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf-8'));
    metadata.image = `ipfs://${cid}`;
    fs.writeFileSync(metadataPath, JSON.stringify(metadata, null, 2));
    console.log('\n3. Updated metadata.json with image URL');
  }

  console.log('\n' + '='.repeat(60));
  console.log('NEXT STEPS:');
  console.log('='.repeat(60));
  console.log('1. Update your GitHub gist metadata.json with the image URL');
  console.log('2. Or upload the updated metadata.json to IPFS');
  console.log('3. Run update-metadata-uri.mjs if using a new metadata URI');

  return cid;
}

// Alternative: Upload to GitHub as raw content
async function printGitHubInstructions() {
  console.log('\n' + '='.repeat(60));
  console.log('ALTERNATIVE: Use GitHub Raw URL');
  console.log('='.repeat(60));
  console.log('\nIf you prefer to use GitHub instead of IPFS:');
  console.log('\n1. Commit cyxtoken.png to your repository');
  console.log('2. Use the raw URL format:');
  console.log('   https://raw.githubusercontent.com/<user>/<repo>/<branch>/cyxtoken.png');
  console.log('\n3. Update metadata.json with this URL');
  console.log('4. Update the gist');
}

async function main() {
  const apiKey = process.argv[2] || process.env.NFT_STORAGE_KEY;

  if (!apiKey) {
    console.log('CYXWIZ Token Image Upload');
    console.log('='.repeat(60));
    console.log('\nNo API key provided.');
    console.log('\nTo upload to IPFS (recommended):');
    console.log('1. Get a free API key from https://nft.storage');
    console.log('2. Run: node upload-image.mjs <YOUR_API_KEY>');
    console.log('   Or set NFT_STORAGE_KEY environment variable');

    printGitHubInstructions();

    console.log('\n' + '='.repeat(60));
    console.log('For quick testing, you can use GitHub raw URL:');
    console.log('='.repeat(60));
    console.log('\nUpdate your gist metadata.json image field to:');
    console.log('https://raw.githubusercontent.com/code3hr/CyxWiz_Claude/GUI/cyxtoken.png');
    console.log('\n(Assuming the image is committed to the GUI branch)');
    process.exit(0);
  }

  await uploadToNFTStorage(apiKey);
}

main().catch(console.error);
