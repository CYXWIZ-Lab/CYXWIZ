// sign_in_presentation.cpp - words and states for the Server Node sign-in
// screen (TOFIX139). See sign_in_presentation.h.
#include "core/sign_in_presentation.h"

#include <algorithm>
#include <cctype>

namespace cyxwiz::servernode::core {

namespace {

std::string Trim(const std::string& text) {
    const auto begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return {};
    const auto end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

std::string Lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

bool Contains(const std::string& text, const std::string& part) {
    return Lower(text).find(Lower(part)) != std::string::npos;
}

// scheme://host[:port][/path] -> scheme, host:port, path
struct UrlParts {
    std::string scheme;
    std::string host_port;
    std::string path;
};

UrlParts SplitUrl(const std::string& url) {
    UrlParts parts;
    std::string rest = url;
    const auto scheme_end = rest.find("://");
    if (scheme_end != std::string::npos) {
        parts.scheme = Lower(rest.substr(0, scheme_end));
        rest = rest.substr(scheme_end + 3);
    }
    const auto slash = rest.find('/');
    if (slash != std::string::npos) {
        parts.path = rest.substr(slash);
        rest = rest.substr(0, slash);
    }
    parts.host_port = rest;
    return parts;
}

const char* kTitle = "CyxWiz Server Node";
const char* kSubtitle = "Sign in to lend this machine to the network";

SignInMessage MessageFor(const SignInInputs& in) {
    SignInMessage m;
    m.shown = in.problem != SignInProblem::None;
    if (!m.shown) return m;
    switch (in.problem) {
        case SignInProblem::Unreachable:
            m.title = "Can't reach the CyxWiz service at " + ServiceHost(in.service_url);
            m.body = "Check that the service is running and that this address is right.";
            m.actions = {SignInAction::TryAgain, SignInAction::ChangeService};
            break;
        case SignInProblem::WrongPassword:
            m.title = "That password doesn't match this email";
            m.body = "Try again, or reset it on the website.";
            m.caps_lock_hint = in.caps_lock;
            break;
        case SignInProblem::NoAccount:
            m.title = "No account with that email";
            m.body = "Check the spelling, or create an account on the website.";
            m.link_word = "create an account";
            m.actions = {SignInAction::CreateAccount};
            break;
        case SignInProblem::Suspended:
            m.title = "This account is suspended";
            m.body = "Sign-in is not possible until it is restored. Contact support on the website.";
            break;
        case SignInProblem::TooManyAttempts:
            m.title = "Too many sign-in attempts";
            m.body = "Wait a minute, then try again.";
            break;
        case SignInProblem::ServiceError:
            m.title = "The CyxWiz service had a problem" +
                      (in.http_status > 0 ? " (HTTP " + std::to_string(in.http_status) + ")" : std::string());
            m.body = "Try again in a moment. If it keeps happening, the node's log has the details.";
            m.actions = {SignInAction::TryAgain};
            break;
        case SignInProblem::Expired:
            m.tone = MessageTone::Info;
            m.title = "Your sign-in expired";
            m.body = "Sign in again to keep lending this machine. Jobs already running were not affected.";
            break;
        case SignInProblem::Other:
            m.title = "Couldn't sign in";
            m.body = "Try again. The node's log has the details.";
            m.actions = {SignInAction::TryAgain};
            break;
        case SignInProblem::None:
            break;
    }
    return m;
}

const char* ProblemName(SignInProblem problem) {
    switch (problem) {
        case SignInProblem::Unreachable: return "service unreachable";
        case SignInProblem::WrongPassword: return "wrong password";
        case SignInProblem::NoAccount: return "no account";
        case SignInProblem::Suspended: return "account suspended";
        case SignInProblem::TooManyAttempts: return "too many attempts";
        case SignInProblem::ServiceError: return "service error";
        case SignInProblem::Expired: return "session expired";
        case SignInProblem::Other: return "sign-in failed";
        case SignInProblem::None: break;
    }
    return "";
}

}  // namespace

SignInCard BuildSignInCard(const SignInInputs& in) {
    SignInCard card;
    const bool busy = in.phase == SignInPhase::SigningIn;
    card.title = kTitle;
    card.subtitle = kSubtitle;
    card.service_prefix = "Signing in to";
    card.service_host = ServiceHost(in.service_url);
    card.change_label = "Change";
    card.change_enabled = !busy;
    card.email_label = "Email";
    card.email_placeholder = "you@example.com";
    card.password_label = "Password";
    card.fields_enabled = !busy;
    card.button_busy = busy;
    card.button_label = busy ? "Signing in\xE2\x80\xA6" : "Sign in";
    card.button_enabled = !busy && !Trim(in.email).empty() && in.password_filled;
    if (busy) card.busy_note = "Checking your password with the CyxWiz service. This takes a few seconds.";
    card.create_prefix = "New to CyxWiz?";
    card.create_link = "Create an account";
    card.create_enabled = !busy;
    card.offline_link = "Work offline";
    card.offline_note = "Local training only. Nothing is lent and no one can reserve this machine.";
    card.offline_enabled = !busy;
    card.footer = std::string(kTitle) + (in.version.empty() ? "" : " " + in.version) +
                  (in.daemon_connected ? " \xC2\xB7 the daemon is connected" : " \xC2\xB7 the daemon is not running");
    card.message = busy ? SignInMessage{} : MessageFor(in);
    if (!busy) card.focus = Trim(in.email).empty() ? SignInFocus::Email : SignInFocus::Password;
    if (in.problem != SignInProblem::None) {
        card.log_line = std::string("Sign-in: ") + ProblemName(in.problem);
        if (in.http_status > 0) card.log_line += " (HTTP " + std::to_string(in.http_status) + ")";
        if (!in.problem_detail.empty()) card.log_line += ": " + in.problem_detail;
    }
    return card;
}

SignInProblem ClassifySignInError(int http_status, const std::string& error_text) {
    if (Contains(error_text, "session expired") || Contains(error_text, "sign-in expired")) {
        return SignInProblem::Expired;
    }
    if (http_status == 0) {
        return error_text.empty() ? SignInProblem::None : SignInProblem::Unreachable;
    }
    if (http_status == 401) {
        return Contains(error_text, "no account") || Contains(error_text, "not found") ? SignInProblem::NoAccount
                                                                                          : SignInProblem::WrongPassword;
    }
    if (http_status == 404) return SignInProblem::NoAccount;
    if (http_status == 403) return SignInProblem::Suspended;
    if (http_status == 429) return SignInProblem::TooManyAttempts;
    if (http_status >= 500) return SignInProblem::ServiceError;
    if (http_status >= 200 && http_status < 300) return SignInProblem::None;
    return SignInProblem::Other;
}

std::string ServiceHost(const std::string& api_url) {
    const auto parts = SplitUrl(Trim(api_url));
    if (parts.host_port.empty()) return Trim(api_url);
    // Default ports are not worth showing.
    const std::string& hp = parts.host_port;
    if (parts.scheme == "https" && hp.size() > 4 && hp.compare(hp.size() - 4, 4, ":443") == 0) {
        return hp.substr(0, hp.size() - 4);
    }
    if (parts.scheme == "http" && hp.size() > 3 && hp.compare(hp.size() - 3, 3, ":80") == 0) {
        return hp.substr(0, hp.size() - 3);
    }
    return hp;
}

std::string WebsiteOf(const std::string& api_url) {
    const auto parts = SplitUrl(Trim(api_url));
    if (parts.host_port.empty()) return {};
    return (parts.scheme.empty() ? std::string("http") : parts.scheme) + "://" + parts.host_port;
}

ServiceAddressCheck CheckServiceAddress(const std::string& typed) {
    ServiceAddressCheck check;
    const std::string text = Trim(typed);
    const char* example = "Enter an address like http://192.168.1.116:3002/api";
    if (text.empty()) {
        check.message = example;
        return check;
    }
    if (text.find_first_of(" \t") != std::string::npos) {
        check.message = std::string("The address has a space in it. ") + example;
        return check;
    }
    const auto parts = SplitUrl(text);
    if (parts.scheme != "http" && parts.scheme != "https") {
        check.message = std::string("The address must start with http:// or https://. ") + example;
        return check;
    }
    if (parts.host_port.empty() || parts.host_port.front() == ':') {
        check.message = std::string("The address has no host. ") + example;
        return check;
    }
    const auto colon = parts.host_port.find(':');
    if (colon != std::string::npos) {
        const std::string port = parts.host_port.substr(colon + 1);
        const bool digits = !port.empty() && std::all_of(port.begin(), port.end(),
                                                         [](unsigned char c) { return std::isdigit(c) != 0; });
        if (!digits || std::stoi(port) < 1 || std::stoi(port) > 65535) {
            check.message = std::string("The port must be a number from 1 to 65535. ") + example;
            return check;
        }
    }
    std::string path = parts.path;
    while (path.size() > 1 && path.back() == '/') path.pop_back();
    if (path.empty() || path == "/") path = "/api";
    check.ok = true;
    check.normalized = parts.scheme + "://" + parts.host_port + path;
    return check;
}

ServiceDialog BuildServiceDialog(const std::string& typed_address, ServiceReach reach) {
    ServiceDialog dialog;
    dialog.title = "Sign-in service";
    dialog.explanation =
        "The CyxWiz website this node signs in with. The central server it reports to is set separately, in "
        "Settings.";
    dialog.address_label = "Address";
    dialog.examples = {{"CyxWiz (production)", "https://cyxwiz.com/api"},
                       {"This PC (local stack)", "http://127.0.0.1:3002/api"},
                       {"Another PC on the network", "http://192.168.1.116:3002/api"}};
    const auto check = CheckServiceAddress(typed_address);
    dialog.address_problem = check.ok ? "" : check.message;
    dialog.reach = check.ok ? reach : ServiceReach::Unknown;
    switch (dialog.reach) {
        case ServiceReach::Reachable: dialog.reach_line = "Reachable \xC2\xB7 CyxWiz API"; break;
        case ServiceReach::Unreachable:
            dialog.reach_line = "Not reachable \xC2\xB7 check the address and that the service is running";
            break;
        case ServiceReach::Checking: dialog.reach_line = "Checking\xE2\x80\xA6"; break;
        case ServiceReach::Unknown: break;
    }
    dialog.note =
        "\"Create an account\" opens the website of this address. Saved in the node's config; the daemon uses it to "
        "renew the sign-in.";
    dialog.cancel_label = "Cancel";
    dialog.use_label = "Use this service";
    dialog.use_enabled = check.ok;
    return dialog;
}

OfflineConfirm BuildOfflineConfirm() {
    OfflineConfirm confirm;
    confirm.title = "Work offline?";
    confirm.lines = {"You can train on this machine from the local Engine and manage its models.",
                     "Nothing is lent: no one can reserve it, it earns nothing, and it does not appear in the "
                     "network's node list.",
                     "Sign in later from Account to go online."};
    confirm.back_label = "Back to sign-in";
    confirm.confirm_label = "Work offline";
    return confirm;
}

ProgressCard BuildProgressCard(const ProgressInputs& in) {
    ProgressCard card;
    const std::string who = !in.user_name.empty() ? in.user_name : (!in.user_email.empty() ? in.user_email : "back");
    card.title = "Welcome, " + who;
    card.subtitle = "Getting this machine onto the network";

    ProgressStep signed_in;
    signed_in.state = StepState::Done;
    signed_in.title = "Signed in";
    signed_in.detail = (in.user_email.empty() ? std::string() : in.user_email + " \xC2\xB7 ") +
                       "stays signed in; renewed in the background";

    ProgressStep registering;
    registering.state = in.registering;
    registering.title = "Registering this machine";
    switch (in.registering) {
        case StepState::Running: registering.detail = "Listing its devices with your account"; break;
        case StepState::Done:
            registering.detail = in.node_name.empty() ? "Listed with your account" : "Listed with your account as " + in.node_name;
            break;
        case StepState::Failed: registering.detail = "Not registered yet"; break;
        case StepState::Pending: registering.detail = "Waits for the sign-in"; break;
    }

    ProgressStep choose;
    choose.state = StepState::Pending;
    choose.title = "Choose what to lend";
    choose.detail = "Opens the Allocation page: pick devices, memory and cores, then Apply to go live";

    card.steps = {signed_in, registering, choose};
    card.footnote = "This card shows for a few seconds, then the dashboard opens.";

    if (in.registering == StepState::Failed) {
        card.failed = true;
        card.failure_title = "Couldn't register this machine";
        if (in.register_unreachable) {
            card.failure_detail = "The CyxWiz service couldn't be reached.";
        } else {
            card.failure_detail = "The CyxWiz service didn't accept it" +
                                  (in.register_error.empty() ? std::string(".") : ": " + in.register_error + ".");
        }
        card.retry_label = "Try again";
        card.continue_label = "Continue to the dashboard";
        card.failure_note = "You stay signed in; the node tries again every minute while it runs.";
        card.footnote.clear();
    }
    return card;
}

}  // namespace cyxwiz::servernode::core
