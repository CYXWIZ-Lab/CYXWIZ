# CYXWIZ Token Integration Guide

## Table of Contents

1. [Solana vs CYXWIZ Relationship](#solana-vs-cyxwiz-relationship)
2. [Wallet Compatibility](#wallet-compatibility)
3. [Token Account Creation](#token-account-creation)
4. [Token Metadata](#token-metadata)
5. [Integration Examples](#integration-examples)

---

## Solana vs CYXWIZ Relationship

### Common Misconception

Many people initially think Solana and CYXWIZ are competing or separate tokens. **They are not.**

### The Actual Relationship

```
┌─────────────────────────────────────────────────────────────────┐
│                    SOLANA BLOCKCHAIN                             │
│                    (The Network/Infrastructure)                  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                                                          │   │
│  │   SOL (Native Currency)     CYXWIZ (SPL Token)          │   │
│  │   - Pays transaction fees   - CyxWiz platform currency   │   │
│  │   - Like ETH on Ethereum    - Like USDC on Ethereum     │   │
│  │                                                          │   │
│  │   Other SPL Tokens on Solana:                           │   │
│  │   - USDC, USDT, BONK, RAY, JUP, etc.                   │   │
│  │                                                          │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Key Definitions

| Term | What It Is | Real-World Analogy |
|------|------------|-------------------|
| **Solana** | Blockchain network | The highway system |
| **SOL** | Native currency of Solana | Gas for your car |
| **SPL Token** | Token standard on Solana | Type of vehicle allowed on highway |
| **CYXWIZ** | SPL token on Solana | Your specific car on the highway |
| **Solana Wallet** | Holds SOL + all SPL tokens | Your garage (holds all vehicles) |

### One Wallet, Multiple Tokens

When a user has a Solana wallet (Phantom, Solflare, etc.), that single wallet can hold:

```
┌─────────────────────────────────────────┐
│         USER'S PHANTOM WALLET           │
│                                         │
│   Address: 7xKXn8mV9rTp2QjH4bLf...      │
│                                         │
│   ┌─────────────────────────────────┐   │
│   │  SOL:      2.5 SOL              │   │  ← Native currency (for fees)
│   │  CYXWIZ:   1,250 CYXWIZ         │   │  ← CyxWiz platform token
│   │  USDC:     100 USDC             │   │  ← Stablecoin
│   │  BONK:     1,000,000 BONK       │   │  ← Meme token
│   │  JUP:      50 JUP               │   │  ← Jupiter token
│   └─────────────────────────────────┘   │
│                                         │
│   All tokens share the SAME address     │
└─────────────────────────────────────────┘
```

### Why SOL is Always Needed

Every transaction on Solana requires a small SOL fee (gas):

| Action | SOL Fee (approx) | USD Equivalent |
|--------|------------------|----------------|
| Transfer CYXWIZ | 0.000005 SOL | ~$0.001 |
| Create token account | 0.002 SOL | ~$0.40 |
| Swap on DEX | 0.0001 SOL | ~$0.02 |
| Smart contract call | 0.0001-0.001 SOL | ~$0.02-0.20 |

**Important:** Users need SOL in their wallet to do anything, even if they're only using CYXWIZ.

### How Users Acquire CYXWIZ

```
┌──────────────────────────────────────────────────────────────┐
│                  HOW TO GET CYXWIZ                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Option 1: Buy on Decentralized Exchange (DEX)              │
│  ┌────────┐      ┌─────────────┐      ┌─────────┐           │
│  │  SOL   │ ───▶ │   Jupiter   │ ───▶ │ CYXWIZ  │           │
│  │  USDC  │      │   Raydium   │      │         │           │
│  └────────┘      └─────────────┘      └─────────┘           │
│                                                              │
│  Option 2: Earn on CyxWiz Platform                          │
│  ┌────────────────┐      ┌─────────────────┐                │
│  │ Run Server Node│ ───▶ │ CYXWIZ Rewards  │                │
│  │ Share ML Models│      │                 │                │
│  └────────────────┘      └─────────────────┘                │
│                                                              │
│  Option 3: Buy with Fiat (Future Feature)                   │
│  ┌────────┐      ┌─────────────┐      ┌─────────┐           │
│  │  USD   │ ───▶ │   Payment   │ ───▶ │ CYXWIZ  │           │
│  │  Card  │      │   Gateway   │      │         │           │
│  └────────┘      └─────────────┘      └─────────┘           │
│                                                              │
│  Option 4: Receive from Another User                        │
│  ┌────────────────┐      ┌─────────────────┐                │
│  │  Friend sends  │ ───▶ │ Your wallet     │                │
│  │  CYXWIZ        │      │ receives CYXWIZ │                │
│  └────────────────┘      └─────────────────┘                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Complete User Flow Example

```
1. USER SETUP
   - User creates Phantom wallet
   - Buys SOL on Coinbase/Binance
   - Transfers SOL to Phantom
   - Wallet now has: 5 SOL

2. GET CYXWIZ
   - User goes to Jupiter (jup.ag)
   - Swaps 2 SOL for 100 CYXWIZ
   - Wallet now has: 3 SOL + 100 CYXWIZ

3. CONNECT TO CYXWIZ PLATFORM
   - User visits cyxwiz.com
   - Clicks "Connect Wallet"
   - Phantom prompts: "Connect to cyxwiz.com?"
   - User approves
   - Platform sees wallet address and CYXWIZ balance

4. DEPOSIT TO PLATFORM
   - User clicks "Deposit" on wallet page
   - Enters amount: 50 CYXWIZ
   - Phantom prompts: "Approve transaction?"
   - User signs transaction
   - SOL fee: ~0.000005 SOL deducted
   - 50 CYXWIZ transfers to platform custody wallet
   - Platform credits Funding account: 50 CYXWIZ

5. USE PLATFORM SERVICES
   - User transfers 30 CYXWIZ: Funding → Spot (internal, free)
   - User runs ML training job: costs 20 CYXWIZ
   - Spot balance: 30 → 10 CYXWIZ

6. EARN REWARDS
   - User runs Server Node software
   - Contributes GPU for 24 hours
   - Earns 15 CYXWIZ in rewards
   - Earn account: 0 → 15 CYXWIZ

7. WITHDRAW
   - User transfers: Earn → Funding (internal, free)
   - User clicks "Withdraw" 20 CYXWIZ
   - Platform sends from custody → user's Phantom wallet
   - SOL fee: ~0.000005 SOL (paid by platform or user)
   - Wallet now has: ~3 SOL + 70 CYXWIZ
```

---

## Wallet Compatibility

### Do Wallets Need to "Accept" CYXWIZ?

**No.** Solana wallets automatically support ALL SPL tokens without any approval or listing process.

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    PHANTOM WALLET                            │
│                                                              │
│  Automatically supports ANY token created on Solana         │
│  - No approval process required                             │
│  - No listing fee                                           │
│  - Works immediately after token creation                   │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  "Known" Tokens (displays name & logo):             │    │
│  │  ✓ SOL, USDC, USDT, BONK, RAY, JUP                 │    │
│  │  (Listed in token registries with metadata)         │    │
│  │                                                     │    │
│  │  "Unknown" Tokens (displays address only):          │    │
│  │  ? CYXw...ABC123 (new tokens without metadata)     │    │
│  │  (Works perfectly, just shows mint address)         │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Supported Wallets

All Solana wallets support CYXWIZ automatically:

| Wallet | Type | Auto-Support | Notes |
|--------|------|--------------|-------|
| **Phantom** | Browser Extension | ✅ Yes | Most popular |
| **Solflare** | Browser/Mobile | ✅ Yes | Full-featured |
| **Backpack** | Browser Extension | ✅ Yes | xNFT support |
| **Ledger** | Hardware | ✅ Yes | Cold storage |
| **Torus** | Social Login | ✅ Yes | Email/Google login |
| **Coinbase Wallet** | Browser/Mobile | ✅ Yes | CEX integration |
| **Trust Wallet** | Mobile | ✅ Yes | Multi-chain |

### What Users See

**Before token metadata is added:**
```
┌─────────────────────────────────────┐
│  Phantom Wallet                     │
│                                     │
│  SOL              2.500             │
│  USDC             100.00            │
│  CYXw...BC123     500.00   ← Shows  │
│  (Unknown Token)           mint     │
│                            address  │
└─────────────────────────────────────┘
```

**After token metadata is added:**
```
┌─────────────────────────────────────┐
│  Phantom Wallet                     │
│                                     │
│  SOL              2.500             │
│  USDC             100.00            │
│  🟣 CYXWIZ        500.00   ← Shows  │
│  CyxWiz Token              name &   │
│                            logo     │
└─────────────────────────────────────┘
```

### Key Points

| Question | Answer |
|----------|--------|
| Does Phantom need to approve CYXWIZ? | **No** - automatic support |
| Can users receive CYXWIZ immediately? | **Yes** - works instantly |
| Will it show the name "CYXWIZ"? | Only **after** metadata is added |
| Will users see the logo? | Only **after** metadata is added |
| Can users send/receive without metadata? | **Yes** - just shows the address |
| Does functionality change with metadata? | **No** - only appearance changes |

---

## Token Account Creation

### Associated Token Accounts (ATA)

On Solana, each token a user holds requires a separate "Token Account":

```
┌────────────────────────────────────────────────────────────┐
│  User's Wallet Address: 7xKXn8mV9rTp2QjH4bLfE6sC...       │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  Main Account (Native SOL)                           │ │
│  │  Balance: 2.5 SOL                                    │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  Associated Token Account (CYXWIZ)                   │ │
│  │  Token Mint: CYXwiz...ABC                            │ │
│  │  Balance: 500 CYXWIZ                                 │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  Associated Token Account (USDC)                     │ │
│  │  Token Mint: EPjFWdd5...                             │ │
│  │  Balance: 100 USDC                                   │ │
│  └──────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
```

### When Token Accounts Are Created

```
First time receiving a token:

1. User has wallet but no CYXWIZ token account
2. Someone sends CYXWIZ to user
3. Transaction automatically creates ATA
4. Rent cost: ~0.002 SOL (usually paid by sender)
5. CYXWIZ arrives in new token account
```

### ATA Address Derivation

Token accounts are deterministically derived:

```typescript
import { getAssociatedTokenAddress } from "@solana/spl-token";

// Given:
const walletAddress = "7xKXn8mV9rTp2QjH4bLfE6sCw5zDg3uYnKaR1W9Fq2Xy";
const cyxwizMint = "CYXwiz...ABC123";

// The ATA address is always the same for this wallet + token combination
const ataAddress = await getAssociatedTokenAddress(
  new PublicKey(cyxwizMint),
  new PublicKey(walletAddress)
);
// Result: deterministic address like "ATAxyz...789"
```

---

## Token Metadata

### Why Add Metadata?

Metadata makes CYXWIZ display nicely in wallets and explorers:

| Without Metadata | With Metadata |
|-----------------|---------------|
| Shows mint address | Shows "CYXWIZ" |
| No logo | Shows logo |
| "Unknown Token" label | Shows "CyxWiz Token" |
| Hard to identify | Easy to identify |

### Metaplex Token Metadata Standard

The standard way to add metadata on Solana:

```typescript
// Metadata structure
{
  name: "CyxWiz Token",
  symbol: "CYXWIZ",
  uri: "https://cyxwiz.com/token-metadata.json",
  sellerFeeBasisPoints: 0,
  creators: null,
  collection: null,
  uses: null
}

// The URI points to a JSON file:
// https://cyxwiz.com/token-metadata.json
{
  "name": "CyxWiz Token",
  "symbol": "CYXWIZ",
  "description": "Native token of the CyxWiz decentralized ML compute platform",
  "image": "https://cyxwiz.com/logo.png",
  "external_url": "https://cyxwiz.com",
  "attributes": []
}
```

### Adding Metadata (Using Metaplex)

```bash
# Install Metaplex CLI
npm install -g @metaplex-foundation/sugar

# Or use the JS SDK
npm install @metaplex-foundation/js
```

```typescript
import { Metaplex } from "@metaplex-foundation/js";

const metaplex = Metaplex.make(connection).use(walletAdapterIdentity(wallet));

await metaplex.nfts().createSft({
  uri: "https://cyxwiz.com/token-metadata.json",
  name: "CyxWiz Token",
  symbol: "CYXWIZ",
  sellerFeeBasisPoints: 0,
  useExistingMint: new PublicKey("YOUR_TOKEN_MINT_ADDRESS"),
});
```

### Token List Registration

For DEX and wallet visibility, register on token lists:

**Jupiter Token List (Recommended):**
- Most DEXs use Jupiter's list
- Submit PR to: https://github.com/jup-ag/token-list

**Solana Token List (Legacy):**
- Still used by some services
- Submit PR to: https://github.com/solana-labs/token-list

---

## Integration Examples

### Reading CYXWIZ Balance

```typescript
import { Connection, PublicKey } from "@solana/web3.js";
import { getAccount, getAssociatedTokenAddress } from "@solana/spl-token";

const connection = new Connection("https://api.devnet.solana.com");
const CYXWIZ_MINT = new PublicKey("YOUR_CYXWIZ_MINT_ADDRESS");

async function getCyxwizBalance(walletAddress: string): Promise<number> {
  try {
    const wallet = new PublicKey(walletAddress);

    // Get the Associated Token Account address
    const tokenAccount = await getAssociatedTokenAddress(CYXWIZ_MINT, wallet);

    // Fetch the account data
    const account = await getAccount(connection, tokenAccount);

    // Convert from raw amount (9 decimals) to readable number
    return Number(account.amount) / 1e9;
  } catch (error) {
    // Token account doesn't exist = 0 balance
    return 0;
  }
}

// Usage
const balance = await getCyxwizBalance("7xKXn8mV9rTp2QjH4bLfE6sCw5zDg3uYnKaR1W9Fq2Xy");
console.log(`Balance: ${balance} CYXWIZ`);
```

### Sending CYXWIZ

```typescript
import {
  Connection,
  Keypair,
  PublicKey,
  Transaction,
  sendAndConfirmTransaction,
} from "@solana/web3.js";
import {
  createTransferInstruction,
  getAssociatedTokenAddress,
  createAssociatedTokenAccountInstruction,
  getAccount,
} from "@solana/spl-token";

async function sendCyxwiz(
  connection: Connection,
  sender: Keypair,
  recipientAddress: string,
  amount: number
): Promise<string> {
  const recipient = new PublicKey(recipientAddress);
  const CYXWIZ_MINT = new PublicKey("YOUR_CYXWIZ_MINT_ADDRESS");

  // Get token accounts
  const senderATA = await getAssociatedTokenAddress(CYXWIZ_MINT, sender.publicKey);
  const recipientATA = await getAssociatedTokenAddress(CYXWIZ_MINT, recipient);

  const transaction = new Transaction();

  // Check if recipient has a token account, create if not
  try {
    await getAccount(connection, recipientATA);
  } catch {
    transaction.add(
      createAssociatedTokenAccountInstruction(
        sender.publicKey,  // payer
        recipientATA,      // token account to create
        recipient,         // owner of token account
        CYXWIZ_MINT        // token mint
      )
    );
  }

  // Add transfer instruction
  transaction.add(
    createTransferInstruction(
      senderATA,           // source
      recipientATA,        // destination
      sender.publicKey,    // owner of source
      BigInt(amount * 1e9) // amount in raw units (9 decimals)
    )
  );

  // Send transaction
  const signature = await sendAndConfirmTransaction(connection, transaction, [sender]);
  return signature;
}
```

### Checking if User Has CYXWIZ Token Account

```typescript
async function hasTokenAccount(walletAddress: string): Promise<boolean> {
  try {
    const wallet = new PublicKey(walletAddress);
    const tokenAccount = await getAssociatedTokenAddress(CYXWIZ_MINT, wallet);
    await getAccount(connection, tokenAccount);
    return true;
  } catch {
    return false;
  }
}
```

---

## Summary

### Solana & CYXWIZ Relationship

- **Solana** is the blockchain network (infrastructure)
- **SOL** is Solana's native currency (needed for transaction fees)
- **CYXWIZ** is an SPL token that lives ON Solana
- One Solana wallet can hold SOL + unlimited SPL tokens
- Users need small amounts of SOL to transact with CYXWIZ

### Wallet Compatibility

- All Solana wallets automatically support CYXWIZ
- No approval or listing process required
- Token metadata is optional (only affects display)
- Users can send/receive CYXWIZ immediately after token creation

### Best Practices

1. Always ensure users have SOL for transaction fees
2. Add token metadata for better UX in wallets
3. Register on Jupiter token list for DEX visibility
4. Handle cases where users don't have a token account yet
5. Use Associated Token Accounts (ATA) for deterministic addresses
