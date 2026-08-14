#include "backend_pack_acquisition.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winhttp.h>
#endif

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cyxwiz::runtime {
namespace {

#ifdef _WIN32

struct InternetHandleCloser {
    void operator()(void* handle) const {
        if (handle) ::WinHttpCloseHandle(handle);
    }
};
using InternetHandle = std::unique_ptr<void, InternetHandleCloser>;

std::wstring WidenUtf8(const std::string& value) {
    if (value.empty()) return {};
    const int size = ::MultiByteToWideChar(
        CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
        static_cast<int>(value.size()), nullptr, 0);
    if (size <= 0) return {};
    std::wstring output(static_cast<std::size_t>(size), L'\0');
    if (::MultiByteToWideChar(
            CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
            static_cast<int>(value.size()), output.data(), size) != size) {
        return {};
    }
    return output;
}

bool QueryHeader(
    HINTERNET request,
    DWORD query,
    std::wstring& output) {
    DWORD bytes = 0;
    if (::WinHttpQueryHeaders(
            request, query, WINHTTP_HEADER_NAME_BY_INDEX, nullptr, &bytes,
            WINHTTP_NO_HEADER_INDEX) || ::GetLastError() !=
                ERROR_INSUFFICIENT_BUFFER) {
        return false;
    }
    std::wstring buffer(bytes / sizeof(wchar_t), L'\0');
    if (!::WinHttpQueryHeaders(
            request, query, WINHTTP_HEADER_NAME_BY_INDEX, buffer.data(),
            &bytes, WINHTTP_NO_HEADER_INDEX)) {
        return false;
    }
    buffer.resize(bytes / sizeof(wchar_t));
    while (!buffer.empty() && buffer.back() == L'\0') buffer.pop_back();
    output = std::move(buffer);
    return true;
}

bool ParseUnsigned(const std::wstring& value, std::uint64_t& output) {
    if (value.empty() ||
        !std::all_of(value.begin(), value.end(), [](wchar_t c) {
            return c >= L'0' && c <= L'9';
        })) {
        return false;
    }
    try {
        std::size_t consumed = 0;
        output = std::stoull(value, &consumed);
        return consumed == value.size();
    } catch (...) {
        return false;
    }
}

std::string WinHttpError(const char* action) {
    return std::string(action) + "; WinHTTP error " +
           std::to_string(::GetLastError());
}

#endif

}  // namespace

HttpsBackendPackArtifactSource::HttpsBackendPackArtifactSource(
    std::string url,
    std::chrono::milliseconds timeout)
    : url_(std::move(url)), timeout_(timeout) {}

std::string HttpsBackendPackArtifactSource::Description() const {
    return url_;
}

bool HttpsBackendPackArtifactSource::TransferFrom(
    std::uint64_t offset,
    std::uint64_t expected_size,
    const BackendPackArtifactChunk& consume,
    const BackendPackArtifactCancelCheck& cancelled,
    std::string& error) {
#ifndef _WIN32
    (void)offset;
    (void)expected_size;
    (void)consume;
    (void)cancelled;
    error = "HTTPS backend-pack acquisition is currently supported on Windows";
    return false;
#else
    if (offset > expected_size || url_.rfind("https://", 0) != 0 ||
        url_.find('#') != std::string::npos || timeout_.count() <= 0 ||
        timeout_.count() > std::numeric_limits<int>::max()) {
        error = "HTTPS artifact source request is invalid";
        return false;
    }
    const auto url = WidenUtf8(url_);
    if (url.empty()) {
        error = "HTTPS artifact URL is not valid UTF-8";
        return false;
    }
    URL_COMPONENTS components{};
    components.dwStructSize = sizeof(components);
    components.dwSchemeLength = static_cast<DWORD>(-1);
    components.dwHostNameLength = static_cast<DWORD>(-1);
    components.dwUrlPathLength = static_cast<DWORD>(-1);
    components.dwExtraInfoLength = static_cast<DWORD>(-1);
    components.dwUserNameLength = static_cast<DWORD>(-1);
    components.dwPasswordLength = static_cast<DWORD>(-1);
    if (!::WinHttpCrackUrl(
            url.data(), static_cast<DWORD>(url.size()), 0, &components) ||
        components.nScheme != INTERNET_SCHEME_HTTPS ||
        components.dwHostNameLength == 0 || components.dwUserNameLength != 0 ||
        components.dwPasswordLength != 0) {
        error = "HTTPS artifact URL cannot be parsed safely";
        return false;
    }
    const std::wstring host(
        components.lpszHostName, components.dwHostNameLength);
    std::wstring request_path = components.dwUrlPathLength == 0
        ? L"/"
        : std::wstring(components.lpszUrlPath, components.dwUrlPathLength);
    if (components.dwExtraInfoLength > 0) {
        request_path.append(
            components.lpszExtraInfo, components.dwExtraInfoLength);
    }

    InternetHandle session(::WinHttpOpen(
        L"CyxWiz Backend Pack Service/1.0",
        WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY, WINHTTP_NO_PROXY_NAME,
        WINHTTP_NO_PROXY_BYPASS, 0));
    if (!session) {
        error = WinHttpError("Cannot initialize HTTPS acquisition");
        return false;
    }
    const int timeout = static_cast<int>(timeout_.count());
    if (!::WinHttpSetTimeouts(
            session.get(), timeout, timeout, timeout, timeout)) {
        error = WinHttpError("Cannot configure HTTPS acquisition timeout");
        return false;
    }
    InternetHandle connection(::WinHttpConnect(
        session.get(), host.c_str(), components.nPort, 0));
    if (!connection) {
        error = WinHttpError("Cannot connect to artifact host");
        return false;
    }
    InternetHandle request(::WinHttpOpenRequest(
        connection.get(), L"GET", request_path.c_str(), nullptr,
        WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, WINHTTP_FLAG_SECURE));
    if (!request) {
        error = WinHttpError("Cannot create HTTPS artifact request");
        return false;
    }
    DWORD redirect_policy = WINHTTP_OPTION_REDIRECT_POLICY_NEVER;
    if (!::WinHttpSetOption(
            request.get(), WINHTTP_OPTION_REDIRECT_POLICY,
            &redirect_policy, sizeof(redirect_policy))) {
        error = WinHttpError("Cannot disable HTTPS redirects");
        return false;
    }
    if (offset > 0) {
        const std::wstring range =
            L"Range: bytes=" + std::to_wstring(offset) + L"-\r\n";
        if (!::WinHttpAddRequestHeaders(
                request.get(), range.c_str(), static_cast<DWORD>(-1),
                WINHTTP_ADDREQ_FLAG_ADD | WINHTTP_ADDREQ_FLAG_REPLACE)) {
            error = WinHttpError("Cannot add HTTPS resume range");
            return false;
        }
    }
    if (cancelled()) {
        error = "Artifact acquisition cancelled";
        return false;
    }
    if (!::WinHttpSendRequest(
            request.get(), WINHTTP_NO_ADDITIONAL_HEADERS, 0,
            WINHTTP_NO_REQUEST_DATA, 0, 0, 0) ||
        !::WinHttpReceiveResponse(request.get(), nullptr)) {
        error = WinHttpError("HTTPS artifact request failed");
        return false;
    }
    DWORD status = 0;
    DWORD status_bytes = sizeof(status);
    if (!::WinHttpQueryHeaders(
            request.get(),
            WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
            WINHTTP_HEADER_NAME_BY_INDEX, &status, &status_bytes,
            WINHTTP_NO_HEADER_INDEX) ||
        status != (offset == 0 ? 200U : 206U)) {
        error = "HTTPS artifact server did not honor the exact transfer request";
        return false;
    }
    std::wstring content_length_text;
    std::uint64_t content_length = 0;
    if (!QueryHeader(
            request.get(), WINHTTP_QUERY_CONTENT_LENGTH,
            content_length_text) ||
        !ParseUnsigned(content_length_text, content_length) ||
        content_length != expected_size - offset) {
        error = "HTTPS Content-Length differs from signed artifact metadata";
        return false;
    }
    if (offset > 0) {
        std::wstring content_range;
        const std::wstring expected_range =
            L"bytes " + std::to_wstring(offset) + L"-" +
            std::to_wstring(expected_size - 1) + L"/" +
            std::to_wstring(expected_size);
        if (!QueryHeader(
                request.get(), WINHTTP_QUERY_CONTENT_RANGE, content_range) ||
            content_range != expected_range) {
            error = "HTTPS Content-Range does not match the resume request";
            return false;
        }
    }

    std::vector<char> buffer(1024 * 1024);
    std::uint64_t transferred = offset;
    for (;;) {
        if (cancelled()) {
            error = "Artifact acquisition cancelled";
            return false;
        }
        DWORD read = 0;
        if (!::WinHttpReadData(
                request.get(), buffer.data(),
                static_cast<DWORD>(buffer.size()), &read)) {
            error = WinHttpError("Cannot read HTTPS artifact response");
            return false;
        }
        if (read == 0) break;
        if (read > expected_size - transferred ||
            !consume(buffer.data(), read, error)) {
            if (error.empty()) error = "HTTPS artifact exceeded signed size";
            return false;
        }
        transferred += read;
    }
    if (transferred != expected_size) {
        error = "HTTPS artifact response ended before its signed size";
        return false;
    }
    return true;
#endif
}

}  // namespace cyxwiz::runtime
