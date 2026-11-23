#pragma once

// The Engine's one CyxCloud gateway connection, where every screen finds it
// (TOFIX164 slice 4b). The Cloud Browser opens it (owns the client) and the
// main window records it here; the Data Input dialog lists datasets through
// it and the Server Connection dialog mints a job's access tokens through it.

#include <string>

namespace network {
class DataStreamClient;
}

namespace cyxwiz::cloud {

// The connected client, or null before the Cloud Browser connected.
void SetClient(network::DataStreamClient* client);
network::DataStreamClient* Client();

// The last connect failure's words ("" once a connect succeeded), for screens
// that say why there is no connection.
void SetLastConnectError(const std::string& text);
std::string LastConnectError();

}  // namespace cyxwiz::cloud
