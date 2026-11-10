#include <catch2/catch_test_macros.hpp>

#include <cyxwiz/utilities.h>

#include <set>
#include <string>

TEST_CASE("Hash utilities preserve standard digest values", "[utilities][hash]") {
    const auto result = cyxwiz::Utilities::HashText("abc", "all");

    REQUIRE(result.success);
    REQUIRE(result.md5_hash == "900150983cd24fb0d6963f7d28e17f72");
    REQUIRE(result.sha1_hash ==
            "a9993e364706816aba3e25717850c26c9cd0d89d");
    REQUIRE(result.sha256_hash ==
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    REQUIRE(result.sha512_hash ==
            "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a"
            "2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f");
}

TEST_CASE("Hash verification is case-insensitive", "[utilities][hash]") {
    REQUIRE(cyxwiz::Utilities::VerifyHash(
        "abc", "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD",
        "sha256"));
    REQUIRE_FALSE(cyxwiz::Utilities::VerifyHash("abc", "deadbeef", "sha256"));
}

TEST_CASE("UUIDs are version 4 and do not repeat across calls", "[utilities][uuid]") {
    std::set<std::string> seen;
    for (int call = 0; call < 2000; ++call) {
        for (const auto& id : cyxwiz::Utilities::GenerateUUIDs(5)) {
            REQUIRE(id.size() == 36);
            REQUIRE(id[8] == '-');
            REQUIRE(id[13] == '-');
            REQUIRE(id[14] == '4');  // version
            REQUIRE(std::string("89ab").find(id[19]) != std::string::npos);  // RFC 4122 variant
            REQUIRE(seen.insert(id).second);
        }
    }
    REQUIRE(seen.size() == 10000);
}

TEST_CASE("A conversion's formula reads in plain digits", "[utilities][units]") {
    const auto km = cyxwiz::Utilities::ConvertUnit(2.0, "kilometer", "meter");
    REQUIRE(km.success);
    REQUIRE(km.output_value == 2000.0);
    REQUIRE(km.formula == "m = km x 1000");
    const auto f = cyxwiz::Utilities::ConvertUnit(100.0, "celsius", "fahrenheit");
    REQUIRE(f.success);
    REQUIRE(f.formula == "\xC2\xB0" "F = \xC2\xB0" "C x 1.8 + 32");
}
