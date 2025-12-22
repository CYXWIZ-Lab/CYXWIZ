// build.rs - gRPC proto compilation
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Compile proto files for gRPC services
    // Enable both server (for incoming gRPC from clients) and client (for outgoing gRPC to Server Nodes)
    tonic_build::configure()
        .build_server(true)
        .build_client(true)  // Enable client code generation for calling Server Nodes
        .compile(
            &[
                "../cyxwiz-protocol/proto/common.proto",
                "../cyxwiz-protocol/proto/job.proto",
                "../cyxwiz-protocol/proto/node.proto",
                "../cyxwiz-protocol/proto/compute.proto",
                "../cyxwiz-protocol/proto/deployment.proto",
                "../cyxwiz-protocol/proto/wallet.proto",
                "../cyxwiz-protocol/proto/reservation.proto",
            ],
            &["../cyxwiz-protocol/proto"],
        )?;
    Ok(())
}
