/**
 * CYXWIZ Token Airdrop Script
 *
 * Usage:
 *   node airdrop.mjs <recipient_address> <amount>
 *   node airdrop.mjs --batch airdrop-list.json
 *
 * Examples:
 *   node airdrop.mjs 7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU 1000
 *   node airdrop.mjs --batch airdrop-list.json
 */

import { execSync } from 'child_process';
import fs from 'fs';

const TOKEN_MINT = 'Az2YZ1hmY5iQ6Gi9rjTPRpNMvcyeYVt1PqjyRSRoNNYi';

function airdrop(recipient, amount) {
  console.log(`\nSending ${amount} CYXWIZ to ${recipient}...`);

  try {
    const cmd = `spl-token transfer ${TOKEN_MINT} ${amount} ${recipient} --fund-recipient`;
    const result = execSync(cmd, { encoding: 'utf-8' });
    console.log('Success!');
    console.log(result);
    return true;
  } catch (error) {
    console.error('Failed:', error.message);
    return false;
  }
}

async function main() {
  const args = process.argv.slice(2);

  if (args.length === 0) {
    console.log('CYXWIZ Token Airdrop');
    console.log('='.repeat(50));
    console.log('\nUsage:');
    console.log('  node airdrop.mjs <recipient_address> <amount>');
    console.log('  node airdrop.mjs --batch <json_file>');
    console.log('\nExamples:');
    console.log('  node airdrop.mjs 7xKXtg...AsU 1000');
    console.log('  node airdrop.mjs --batch airdrop-list.json');
    console.log('\nBatch file format (airdrop-list.json):');
    console.log('  [');
    console.log('    { "address": "7xKXtg...", "amount": 1000 },');
    console.log('    { "address": "9aBcDe...", "amount": 500 }');
    console.log('  ]');
    console.log('\nToken Mint:', TOKEN_MINT);

    // Show current balance
    try {
      const balance = execSync(`spl-token balance ${TOKEN_MINT}`, { encoding: 'utf-8' });
      console.log('Your Balance:', balance.trim(), 'CYXWIZ');
    } catch (e) {
      console.log('Could not fetch balance');
    }

    process.exit(0);
  }

  // Batch mode
  if (args[0] === '--batch') {
    const batchFile = args[1];
    if (!batchFile || !fs.existsSync(batchFile)) {
      console.error('Error: Batch file not found:', batchFile);
      process.exit(1);
    }

    const recipients = JSON.parse(fs.readFileSync(batchFile, 'utf-8'));
    console.log('CYXWIZ Batch Airdrop');
    console.log('='.repeat(50));
    console.log(`Processing ${recipients.length} recipients...\n`);

    let success = 0;
    let failed = 0;

    for (const { address, amount } of recipients) {
      if (airdrop(address, amount)) {
        success++;
      } else {
        failed++;
      }
    }

    console.log('\n' + '='.repeat(50));
    console.log(`Completed: ${success} success, ${failed} failed`);
    return;
  }

  // Single transfer mode
  const recipient = args[0];
  const amount = args[1] || '100';

  if (!recipient || recipient.length < 32) {
    console.error('Error: Invalid recipient address');
    process.exit(1);
  }

  console.log('CYXWIZ Token Airdrop');
  console.log('='.repeat(50));
  console.log('Token Mint:', TOKEN_MINT);
  console.log('Recipient:', recipient);
  console.log('Amount:', amount, 'CYXWIZ');

  airdrop(recipient, amount);
}

main().catch(console.error);
