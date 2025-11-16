//! Generated protobuf code module
//!
//! This module contains all generated protocol buffer types and gRPC service definitions
//! from the cyxwiz-protocol package.

// Re-export all generated protobuf types
pub use pb_inner::*;

/// Inner module to encapsulate the generated protobuf code
mod pb_inner {
    // Include the generated protobuf code from build.rs
    tonic::include_proto!("cyxwiz.protocol");
}
