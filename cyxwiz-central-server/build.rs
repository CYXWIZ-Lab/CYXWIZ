// build.rs - gRPC proto compilation
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Compile proto files for gRPC services
    tonic_build::configure()
        .build_server(true)
        .build_client(false)
        .compile(
            &[
                "../cyxwiz-protocol/proto/common.proto",
                "../cyxwiz-protocol/proto/job.proto",
                "../cyxwiz-protocol/proto/node.proto",
                "../cyxwiz-protocol/proto/compute.proto",
                "../cyxwiz-protocol/proto/deployment.proto",
            ],
            &["../cyxwiz-protocol/proto"],
        )?;
    Ok(())
}
