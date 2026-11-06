#include "server_address.h"

#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <grpcpp/support/channel_arguments.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>

namespace cyxwiz::protocol {
namespace {

std::string Trim(const std::string& s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

bool StartsWith(const std::string& s, const char* prefix) {
    return s.rfind(prefix, 0) == 0;
}

// "host:port", "[::1]:port" -> true and the parts; "host", "[::1]" -> false.
bool SplitPort(const std::string& host_port, std::string* host, std::string* port) {
    const auto close_bracket = host_port.rfind(']');
    const auto colon = host_port.rfind(':');
    if (colon == std::string::npos ||
        (close_bracket != std::string::npos && colon < close_bracket)) {
        *host = host_port;
        port->clear();
        return false;
    }
    *host = host_port.substr(0, colon);
    *port = host_port.substr(colon + 1);
    return true;
}

bool ValidPort(const std::string& port) {
    if (port.empty() || port.size() > 5 ||
        !std::all_of(port.begin(), port.end(), [](unsigned char c) { return std::isdigit(c); })) {
        return false;
    }
    const int n = std::stoi(port);
    return n > 0 && n <= 65535;
}

ServerEndpoints Error(std::string message) {
    ServerEndpoints e;
    e.error = std::move(message);
    return e;
}

std::string Lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

bool EndsWith(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// "a.b.c.d" -> the four octets
bool ParseIpv4(const std::string& s, unsigned out[4]) {
    std::istringstream in(s);
    std::string part;
    int n = 0;
    while (std::getline(in, part, '.')) {
        if (n == 4 || part.empty() || part.size() > 3 ||
            !std::all_of(part.begin(), part.end(), [](unsigned char c) { return std::isdigit(c); })) {
            return false;
        }
        const unsigned v = static_cast<unsigned>(std::stoul(part));
        if (v > 255) return false;
        out[n++] = v;
    }
    return n == 4 && s.back() != '.';
}

bool PrivateIpv4(const unsigned o[4]) {
    return o[0] == 127 || o[0] == 10 || (o[0] == 172 && o[1] >= 16 && o[1] <= 31) ||
           (o[0] == 192 && o[1] == 168) || (o[0] == 169 && o[1] == 254) ||
           (o[0] == 100 && (o[1] & 0xc0) == 64);
}

// The first 16-bit group of an IPv6 literal and whether it is "::1"; an
// "::ffff:a.b.c.d" literal gives its IPv4 part instead.
bool PrivateIpv6(const std::string& ip) {
    if (ip == "::1") return true;
    const auto last_colon = ip.rfind(':');
    if (StartsWith(ip, "::ffff:") && last_colon != std::string::npos) {
        unsigned o[4];
        return ParseIpv4(ip.substr(last_colon + 1), o) && PrivateIpv4(o);
    }
    const auto first_colon = ip.find(':');
    if (first_colon == 0 || first_colon == std::string::npos || first_colon > 4) return false;
    const std::string group = ip.substr(0, first_colon);
    if (!std::all_of(group.begin(), group.end(), [](unsigned char c) { return std::isxdigit(c); })) return false;
    const unsigned first = static_cast<unsigned>(std::stoul(group, nullptr, 16));
    return (first & 0xfe00) == 0xfc00 || (first & 0xffc0) == 0xfe80;
}

std::string StripBrackets(std::string host) {
    host = Trim(host);
    if (!host.empty() && host.front() == '[') host.erase(0, 1);
    if (!host.empty() && host.back() == ']') host.pop_back();
    while (!host.empty() && host.back() == '.') host.pop_back();
    return Lower(host);
}

ServerEndpoints PlainToThePublic(const std::string& typed, const std::string& host) {
    return Error("'" + typed + "' is plain http:// to " + host +
                 ", which is not on this machine or a private network: your sign-in would cross the "
                 "internet unencrypted. Use the server's https:// address.");
}

// A root set that holds no usable certificate: every handshake fails.
constexpr const char* kNoRootVerifies = "-----BEGIN CERTIFICATE-----\n-----END CERTIFICATE-----\n";

std::mutex& CaMutex() {
    static std::mutex m;
    return m;
}
std::string& CaPath() {
    static std::string path;
    return path;
}
bool& CaSet() {
    static bool set = false;
    return set;
}

// The CA's PEM text; "" with `error` set when the file cannot be used.
std::string ReadCaPem(const std::string& path, std::string* error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        *error = "the server's CA file " + path + " cannot be read";
        return {};
    }
    std::ostringstream text;
    text << in.rdbuf();
    std::string pem = text.str();
    if (pem.find("-----BEGIN CERTIFICATE-----") == std::string::npos) {
        *error = "the server's CA file " + path + " holds no PEM certificate";
        return {};
    }
    error->clear();
    return pem;
}

}  // namespace

bool PlainHttpAllowed(const std::string& host_in) {
    const std::string host = StripBrackets(host_in);
    if (host.empty()) return false;
    unsigned o[4];
    if (ParseIpv4(host, o)) return PrivateIpv4(o);
    if (host.find(':') != std::string::npos) return PrivateIpv6(host);
    if (host == "localhost" || EndsWith(host, ".localhost")) return true;
    if (host.find('.') == std::string::npos) return true;
    for (const char* suffix : {".local", ".lan", ".internal", ".home.arpa"}) {
        if (EndsWith(host, suffix)) return true;
    }
    return false;
}

bool IsLoopbackHost(const std::string& host_in) {
    const std::string host = StripBrackets(host_in);
    unsigned o[4];
    return host == "localhost" || EndsWith(host, ".localhost") || host == "::1" ||
           (ParseIpv4(host, o) && o[0] == 127);
}

void SetServerCaFile(const std::string& path) {
    std::lock_guard<std::mutex> lock(CaMutex());
    CaPath() = Trim(path);
    CaSet() = true;
}

std::string ServerCaFile() {
    {
        std::lock_guard<std::mutex> lock(CaMutex());
        if (CaSet() && !CaPath().empty()) return CaPath();
    }
    const char* env = std::getenv("CYXWIZ_CA_FILE");
    return env ? Trim(env) : std::string();
}

std::string ServerCaError() {
    const std::string path = ServerCaFile();
    if (path.empty()) return {};
    std::string error;
    ReadCaPem(path, &error);
    return error;
}

ServerEndpoints ResolveServerAddress(const std::string& typed) {
    std::string address = Trim(typed);
    while (!address.empty() && address.back() == '/') {
        address.pop_back();
    }
    if (address.empty()) {
        return Error("the server address is empty");
    }

    const bool https = StartsWith(address, "https://");
    const bool http = StartsWith(address, "http://");
    const std::string rest = https ? address.substr(8) : http ? address.substr(7) : address;
    if (rest.empty() || rest.find('/') != std::string::npos) {
        return Error("'" + typed + "' is not a server address (no path: https://HOST, http://HOST:PORT or HOST)");
    }

    std::string host, port;
    const bool has_port = SplitPort(rest, &host, &port);
    if (host.empty() || (has_port && !ValidPort(port))) {
        return Error("'" + typed + "' has no usable host or port");
    }

    if (!https && !PlainHttpAllowed(host)) {
        return PlainToThePublic(typed, host);
    }

    ServerEndpoints e;
    if (https || (http && has_port)) {
        // A front door: everything at one origin
        e.front_door = true;
        e.tls = https;
        const std::string grpc_port = has_port ? port : (https ? "443" : "80");
        const std::string target = host + ":" + grpc_port;
        e.central = https ? "https://" + target : target;
        e.gateway = e.central;
        e.api_base_url = (https ? "https://" : "http://") + rest + "/api";
        return e;
    }
    if (has_port) {
        return Error("'" + typed +
                     "': give the server's host name or IP, or a front door as http://HOST:PORT or https://HOST");
    }
    // No front door: the three services on their own ports
    e.central = host + ":" + std::to_string(kCentralGrpcPort);
    e.gateway = host + ":" + std::to_string(kGatewayGrpcPort);
    e.api_base_url = "http://" + host + ":" + std::to_string(kApiPort) + "/api";
    return e;
}

GrpcTarget ParseGrpcAddress(const std::string& typed) {
    std::string address = Trim(typed);
    while (!address.empty() && address.back() == '/') {
        address.pop_back();
    }
    GrpcTarget t;
    if (StartsWith(address, "https://")) {
        t.tls = true;
        address = address.substr(8);
        std::string host, port;
        if (!SplitPort(address, &host, &port)) {
            address += ":443";
        }
    } else if (StartsWith(address, "http://")) {
        address = address.substr(7);
        std::string host, port;
        if (!SplitPort(address, &host, &port)) {
            address += ":80";
        }
    }
    t.target = address;
    return t;
}

std::shared_ptr<grpc::Channel> CreateServerChannel(const std::string& address) {
    return CreateServerChannel(address, grpc::ChannelArguments());
}

std::shared_ptr<grpc::Channel> CreateServerChannel(const std::string& address,
                                                   const grpc::ChannelArguments& args) {
    const GrpcTarget t = ParseGrpcAddress(address);
    if (!t.tls) {
        return grpc::CreateCustomChannel(t.target, grpc::InsecureChannelCredentials(), args);
    }
    grpc::SslCredentialsOptions options;  // empty roots: the system's
    const std::string ca_path = ServerCaFile();
    if (!ca_path.empty()) {
        std::string error;
        options.pem_root_certs = ReadCaPem(ca_path, &error);
        // A CA that cannot be read must not fall back to the public roots
        // (they were not what the user asked to trust): no root verifies.
        if (options.pem_root_certs.empty()) options.pem_root_certs = kNoRootVerifies;
    }
    return grpc::CreateCustomChannel(t.target, grpc::SslCredentials(options), args);
}

}  // namespace cyxwiz::protocol
