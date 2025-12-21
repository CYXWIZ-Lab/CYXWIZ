//! JobEscrow smart contract instruction builders
//!
//! This module provides instruction builders for interacting with the
//! JobEscrow Anchor program deployed at: DefY4GG33pAgBJqwPKDSKbPbCKmoCcN8oymvHhzsp2dA
//!
//! Contract Instructions:
//! - CreateEscrow: Lock user funds for a job
//! - ReleasePayment: Release 90% to node, 10% to platform
//! - Refund: Return full amount to user if job fails

use borsh::BorshSerialize;
use sha2::{Digest, Sha256};
use solana_sdk::{
    instruction::{AccountMeta, Instruction},
    pubkey::Pubkey,
    system_program,
};
use spl_token;

/// JobEscrow Program ID on Solana Devnet
pub const JOB_ESCROW_PROGRAM_ID: &str = "DefY4GG33pAgBJqwPKDSKbPbCKmoCcN8oymvHhzsp2dA";

/// Compute Anchor instruction discriminator from instruction name
/// Formula: sha256("global:<instruction_name>")[0..8]
fn anchor_discriminator(instruction_name: &str) -> [u8; 8] {
    let preimage = format!("global:{}", instruction_name);
    let hash = Sha256::digest(preimage.as_bytes());
    let mut discriminator = [0u8; 8];
    discriminator.copy_from_slice(&hash[..8]);
    discriminator
}

/// Create Escrow instruction arguments (Anchor/Borsh serialized)
#[derive(BorshSerialize)]
struct CreateEscrowArgs {
    job_id: u64,
    amount: u64,
    node_pubkey: Pubkey,
}

/// Escrow account state (matching the Anchor program)
#[derive(Debug, Clone)]
pub struct EscrowState {
    pub job_id: u64,
    pub user: Pubkey,
    pub node: Pubkey,
    pub amount: u64,
    pub platform_fee_percentage: u8,
    pub status: EscrowStatus,
    pub created_at: i64,
    pub completed_at: Option<i64>,
    pub bump: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscrowStatus {
    Pending = 0,
    Released = 1,
    Refunded = 2,
}

/// Derive the escrow PDA address for a given job ID
///
/// Seeds: ["escrow", job_id.to_le_bytes()]
pub fn find_escrow_address(program_id: &Pubkey, job_id: u64) -> (Pubkey, u8) {
    Pubkey::find_program_address(
        &[b"escrow", &job_id.to_le_bytes()],
        program_id,
    )
}

/// Build CreateEscrow instruction
///
/// Creates an escrow account that locks user funds until job completion.
///
/// # Arguments
/// * `program_id` - JobEscrow program ID
/// * `user` - User's wallet (signer, pays for account creation)
/// * `user_token_account` - User's SPL token account (source of funds)
/// * `escrow_token_account` - Escrow's SPL token account (destination)
/// * `job_id` - Unique job identifier
/// * `amount` - Amount of tokens to escrow
/// * `node_pubkey` - Compute node's public key (will receive payment)
///
/// # Account layout (matching Anchor CreateEscrow context):
/// 0. escrow (writable) - PDA
/// 1. user (signer, writable) - Payer
/// 2. user_token_account (writable) - Source
/// 3. escrow_token_account (writable) - Destination
/// 4. token_program (readonly)
/// 5. system_program (readonly)
pub fn create_escrow_instruction(
    program_id: &Pubkey,
    user: &Pubkey,
    user_token_account: &Pubkey,
    escrow_token_account: &Pubkey,
    job_id: u64,
    amount: u64,
    node_pubkey: &Pubkey,
) -> Instruction {
    let (escrow_pda, _bump) = find_escrow_address(program_id, job_id);

    // Build instruction data: discriminator + args
    let discriminator = anchor_discriminator("create_escrow");
    let args = CreateEscrowArgs {
        job_id,
        amount,
        node_pubkey: *node_pubkey,
    };

    let mut data = discriminator.to_vec();
    args.serialize(&mut data).expect("Failed to serialize args");

    Instruction {
        program_id: *program_id,
        accounts: vec![
            AccountMeta::new(escrow_pda, false),              // escrow PDA (writable)
            AccountMeta::new(*user, true),                    // user (signer, writable)
            AccountMeta::new(*user_token_account, false),     // user_token_account (writable)
            AccountMeta::new(*escrow_token_account, false),   // escrow_token_account (writable)
            AccountMeta::new_readonly(spl_token::id(), false), // token_program
            AccountMeta::new_readonly(system_program::id(), false), // system_program
        ],
        data,
    }
}

/// Build ReleasePayment instruction
///
/// Releases escrowed funds to the compute node (90%) and platform (10%).
/// Can only be called when escrow status is Pending.
///
/// # Arguments
/// * `program_id` - JobEscrow program ID
/// * `job_id` - Job identifier (used to derive escrow PDA)
/// * `escrow_token_account` - Escrow's SPL token account
/// * `node_token_account` - Node's SPL token account (receives 90%)
/// * `platform_token_account` - Platform's SPL token account (receives 10%)
/// * `authority` - Transaction signer (Central Server)
///
/// # Account layout (matching Anchor ReleasePayment context):
/// 0. escrow (writable) - PDA
/// 1. escrow_token_account (writable)
/// 2. node_token_account (writable)
/// 3. platform_token_account (writable)
/// 4. authority (signer)
/// 5. token_program (readonly)
pub fn release_payment_instruction(
    program_id: &Pubkey,
    job_id: u64,
    escrow_token_account: &Pubkey,
    node_token_account: &Pubkey,
    platform_token_account: &Pubkey,
    authority: &Pubkey,
) -> Instruction {
    let (escrow_pda, _bump) = find_escrow_address(program_id, job_id);

    // ReleasePayment has no additional args, just discriminator
    let discriminator = anchor_discriminator("release_payment");
    let data = discriminator.to_vec();

    Instruction {
        program_id: *program_id,
        accounts: vec![
            AccountMeta::new(escrow_pda, false),               // escrow PDA (writable)
            AccountMeta::new(*escrow_token_account, false),    // escrow_token_account (writable)
            AccountMeta::new(*node_token_account, false),      // node_token_account (writable)
            AccountMeta::new(*platform_token_account, false),  // platform_token_account (writable)
            AccountMeta::new_readonly(*authority, true),       // authority (signer)
            AccountMeta::new_readonly(spl_token::id(), false), // token_program
        ],
        data,
    }
}

/// Build Refund instruction
///
/// Refunds the full escrowed amount back to the user.
/// Can only be called when escrow status is Pending.
///
/// # Arguments
/// * `program_id` - JobEscrow program ID
/// * `job_id` - Job identifier (used to derive escrow PDA)
/// * `escrow_token_account` - Escrow's SPL token account (source)
/// * `user_token_account` - User's SPL token account (destination)
/// * `authority` - Transaction signer
///
/// # Account layout (matching Anchor Refund context):
/// 0. escrow (writable) - PDA
/// 1. escrow_token_account (writable)
/// 2. user_token_account (writable)
/// 3. authority (signer)
/// 4. token_program (readonly)
pub fn refund_instruction(
    program_id: &Pubkey,
    job_id: u64,
    escrow_token_account: &Pubkey,
    user_token_account: &Pubkey,
    authority: &Pubkey,
) -> Instruction {
    let (escrow_pda, _bump) = find_escrow_address(program_id, job_id);

    // Refund has no additional args, just discriminator
    let discriminator = anchor_discriminator("refund");
    let data = discriminator.to_vec();

    Instruction {
        program_id: *program_id,
        accounts: vec![
            AccountMeta::new(escrow_pda, false),               // escrow PDA (writable)
            AccountMeta::new(*escrow_token_account, false),    // escrow_token_account (writable)
            AccountMeta::new(*user_token_account, false),      // user_token_account (writable)
            AccountMeta::new_readonly(*authority, true),       // authority (signer)
            AccountMeta::new_readonly(spl_token::id(), false), // token_program
        ],
        data,
    }
}

/// Parse escrow state from account data
///
/// Anchor accounts have an 8-byte discriminator prefix, followed by
/// the Borsh-serialized struct fields.
pub fn parse_escrow_state(data: &[u8]) -> Option<EscrowState> {
    if data.len() < 8 {
        return None;
    }

    // Skip 8-byte discriminator
    let data = &data[8..];

    // Manual parsing based on Escrow struct layout:
    // job_id: u64 (8 bytes)
    // user: Pubkey (32 bytes)
    // node: Pubkey (32 bytes)
    // amount: u64 (8 bytes)
    // platform_fee_percentage: u8 (1 byte)
    // status: enum (1 byte)
    // created_at: i64 (8 bytes)
    // completed_at: Option<i64> (1 + 8 bytes)
    // bump: u8 (1 byte)

    if data.len() < 8 + 32 + 32 + 8 + 1 + 1 + 8 + 1 + 1 {
        return None;
    }

    let job_id = u64::from_le_bytes(data[0..8].try_into().ok()?);
    let user = Pubkey::try_from(&data[8..40]).ok()?;
    let node = Pubkey::try_from(&data[40..72]).ok()?;
    let amount = u64::from_le_bytes(data[72..80].try_into().ok()?);
    let platform_fee_percentage = data[80];
    let status = match data[81] {
        0 => EscrowStatus::Pending,
        1 => EscrowStatus::Released,
        2 => EscrowStatus::Refunded,
        _ => return None,
    };
    let created_at = i64::from_le_bytes(data[82..90].try_into().ok()?);

    let (completed_at, bump_offset) = if data[90] == 1 {
        let ts = i64::from_le_bytes(data[91..99].try_into().ok()?);
        (Some(ts), 99)
    } else {
        (None, 91)
    };

    let bump = data[bump_offset];

    Some(EscrowState {
        job_id,
        user,
        node,
        amount,
        platform_fee_percentage,
        status,
        created_at,
        completed_at,
        bump,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_anchor_discriminator() {
        // Test discriminator computation
        let disc = anchor_discriminator("create_escrow");
        assert_eq!(disc.len(), 8);

        // Different instructions should have different discriminators
        let disc2 = anchor_discriminator("release_payment");
        assert_ne!(disc, disc2);

        // Same instruction should be deterministic
        let disc3 = anchor_discriminator("create_escrow");
        assert_eq!(disc, disc3);
    }

    #[test]
    fn test_find_escrow_address() {
        let program_id = Pubkey::from_str(JOB_ESCROW_PROGRAM_ID).unwrap();
        let job_id = 12345u64;

        let (pda1, bump1) = find_escrow_address(&program_id, job_id);
        let (pda2, bump2) = find_escrow_address(&program_id, job_id);

        // PDAs should be deterministic
        assert_eq!(pda1, pda2);
        assert_eq!(bump1, bump2);

        // Different job IDs should give different PDAs
        let (pda3, _) = find_escrow_address(&program_id, job_id + 1);
        assert_ne!(pda1, pda3);
    }

    #[test]
    fn test_create_escrow_instruction_format() {
        let program_id = Pubkey::from_str(JOB_ESCROW_PROGRAM_ID).unwrap();
        let user = Pubkey::new_unique();
        let user_token = Pubkey::new_unique();
        let escrow_token = Pubkey::new_unique();
        let node = Pubkey::new_unique();

        let ix = create_escrow_instruction(
            &program_id,
            &user,
            &user_token,
            &escrow_token,
            12345,
            1000,
            &node,
        );

        // Verify account count
        assert_eq!(ix.accounts.len(), 6);

        // Verify instruction data starts with discriminator
        assert!(ix.data.len() > 8);

        // User should be signer
        assert!(ix.accounts[1].is_signer);
    }

    #[test]
    fn test_release_payment_instruction_format() {
        let program_id = Pubkey::from_str(JOB_ESCROW_PROGRAM_ID).unwrap();
        let escrow_token = Pubkey::new_unique();
        let node_token = Pubkey::new_unique();
        let platform_token = Pubkey::new_unique();
        let authority = Pubkey::new_unique();

        let ix = release_payment_instruction(
            &program_id,
            12345,
            &escrow_token,
            &node_token,
            &platform_token,
            &authority,
        );

        // Verify account count
        assert_eq!(ix.accounts.len(), 6);

        // Verify instruction data is just discriminator (8 bytes)
        assert_eq!(ix.data.len(), 8);

        // Authority should be signer
        assert!(ix.accounts[4].is_signer);
    }

    #[test]
    fn test_refund_instruction_format() {
        let program_id = Pubkey::from_str(JOB_ESCROW_PROGRAM_ID).unwrap();
        let escrow_token = Pubkey::new_unique();
        let user_token = Pubkey::new_unique();
        let authority = Pubkey::new_unique();

        let ix = refund_instruction(
            &program_id,
            12345,
            &escrow_token,
            &user_token,
            &authority,
        );

        // Verify account count
        assert_eq!(ix.accounts.len(), 5);

        // Verify instruction data is just discriminator (8 bytes)
        assert_eq!(ix.data.len(), 8);

        // Authority should be signer
        assert!(ix.accounts[3].is_signer);
    }
}
