// sign_in_presentation.h - words and states for the Server Node sign-in screen
// (TOFIX139). Pure data in, data out: no ImGui, no network, no globals, so
// the GUI renders exactly what the test checks.
#pragma once

#include "common/server_address_view.h"

#include <string>
#include <utility>
#include <vector>

namespace cyxwiz::servernode::core {

// ---- the sign-in card -------------------------------------------------------

enum class SignInPhase { Form, SigningIn };

// What went wrong, classified from the web API's answer (never shown raw).
enum class SignInProblem {
    None,
    Unreachable,      // no connection, timeout, the old "Read error"
    WrongPassword,    // HTTP 401 after the password check
    NoAccount,        // HTTP 401/404 before any password check
    Suspended,        // HTTP 403
    TooManyAttempts,  // HTTP 429
    ServiceError,     // HTTP 5xx
    Expired,          // the saved session stopped being accepted
    Other
};

enum class MessageTone { Error, Info };
enum class SignInAction { TryAgain, ChangeService, CreateAccount };
enum class SignInFocus { None, Email, Password };

struct SignInMessage {
    bool shown = false;
    MessageTone tone = MessageTone::Error;
    std::string title;
    std::string body;
    std::string link_word;            // part of body drawn as a link ("create an account")
    std::vector<SignInAction> actions;
    bool caps_lock_hint = false;      // "Caps Lock is on"
};

struct SignInInputs {
    SignInPhase phase = SignInPhase::Form;
    std::string service_url;          // "http://127.0.0.1:3002/api"
    std::string email;
    bool password_filled = false;
    bool caps_lock = false;
    SignInProblem problem = SignInProblem::None;
    int http_status = 0;              // for the service-error title
    std::string problem_detail;       // the API's words, for the log line only
    bool daemon_connected = false;
    std::string version;              // "0.3" ("" when unknown)
};

struct SignInCard {
    std::string title;                // "CyxWiz Server Node"
    std::string subtitle;             // "Sign in to lend this machine to the network"
    std::string service_prefix;       // "Signing in to"
    std::string service_host;         // "127.0.0.1:3002"
    std::string change_label;         // "Change"
    bool change_enabled = true;
    std::string email_label;
    std::string email_placeholder;
    std::string password_label;
    bool fields_enabled = true;
    std::string button_label;         // "Sign in" / "Signing in…"
    bool button_enabled = false;
    bool button_busy = false;
    std::string busy_note;            // under the button while signing in
    std::string create_prefix;        // "New to CyxWiz?"
    std::string create_link;          // "Create an account"
    bool create_enabled = true;
    std::string offline_link;         // "Work offline"
    std::string offline_note;
    bool offline_enabled = true;
    std::string footer;               // "CyxWiz Server Node 0.3 · the daemon is connected"
    SignInMessage message;
    SignInFocus focus = SignInFocus::None;  // where the cursor goes when the screen opens
    std::string log_line;             // what to log for a problem ("" when none)
};

SignInCard BuildSignInCard(const SignInInputs& in);

// The problem behind an AuthResult: the HTTP status decides, the text only
// tells the two 401s apart. status 0 = no answer.
SignInProblem ClassifySignInError(int http_status, const std::string& error_text);

// "http://127.0.0.1:3002/api" -> "127.0.0.1:3002"; "https://cyxwiz.com/api" -> "cyxwiz.com".
std::string ServiceHost(const std::string& api_url);

// The website behind an API address: "https://cyxwiz.com/api" -> "https://cyxwiz.com".
std::string WebsiteOf(const std::string& api_url);

// ---- The CyxWiz server (TOFIX168 step 3b) -----------------------------------
// "Change" on the card opens this: the one server address the node signs in
// at and reports to central through. The rows and their wording come from
// cyxwiz-protocol's server_address_view, as in Settings > Network.

struct ServiceDialog {
    std::string title;
    std::string explanation;
    std::string address_label;
    std::vector<std::pair<std::string, std::string>> examples;  // (what, address)
    cyxwiz::protocol::ServerAddressView view;  // kind of address and one row per service
    std::string note;
    std::string cancel_label;
    std::string use_label;
    bool use_enabled = false;
    std::string address_problem;      // under the field when the address can't be used
};

ServiceDialog BuildServiceDialog(const std::string& typed_address, const cyxwiz::protocol::ServerReach& reach);

// ---- Work offline ------------------------------------------------------------

struct OfflineConfirm {
    std::string title;
    std::vector<std::string> lines;   // the last one is muted
    std::string back_label;
    std::string confirm_label;
};

OfflineConfirm BuildOfflineConfirm();

// ---- After sign-in -----------------------------------------------------------

enum class StepState { Done, Running, Pending, Failed };

struct ProgressStep {
    StepState state = StepState::Pending;
    std::string title;
    std::string detail;
};

struct ProgressInputs {
    std::string user_name;
    std::string user_email;
    StepState registering = StepState::Running;
    std::string node_name;            // when registered
    std::string register_error;       // the API's words when registering failed
    bool register_unreachable = false; // the failure was no connection
};

struct ProgressCard {
    std::string title;                // "Welcome, e2e_user"
    std::string subtitle;
    std::vector<ProgressStep> steps;  // Signed in, Registering, Choose what to lend
    std::string footnote;
    bool failed = false;
    std::string failure_title;
    std::string failure_detail;
    std::string retry_label;
    std::string continue_label;
    std::string failure_note;
};

ProgressCard BuildProgressCard(const ProgressInputs& in);

}  // namespace cyxwiz::servernode::core
