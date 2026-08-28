#pragma once

#include <string>
#include <string_view>

namespace cyxwiz::runtime {

bool AuthorizeGithubReleaseAssetRedirect(
    std::string_view original_url,
    unsigned int response_status,
    std::string_view location,
    std::string& authorized_url,
    std::string& error);

}  // namespace cyxwiz::runtime
