#include "amuquant/canonical_history.hpp"

#include <algorithm>
#include <sstream>
#include <cctype>
#include <unordered_set>
#include <cmath>

namespace amuquant {

namespace {

std::string to_lower_str(const std::string& str) {
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    return lower;
}

bool is_alphanumeric_or_symbol(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_' || c == '.';
}

std::vector<std::string> extract_query_keywords(const std::string& text) {
    static const std::unordered_set<std::string> STOPWORDS = {
        "dan", "di", "ke", "dari", "yang", "ini", "itu", "untuk", "pada", "adalah",
        "apa", "siapa", "berapa", "dimana", "kapan", "bagaimana", "mengapa", "apakah",
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for",
        "of", "with", "by", "what", "who", "which", "where", "how", "many", "project",
        "proyek", "nomor", "versi", "pemilik", "port"
    };

    std::vector<std::string> tokens;
    std::string current;

    for (char c : text) {
        if (is_alphanumeric_or_symbol(c)) {
            current.push_back(c);
        } else if (!current.empty()) {
            std::string lower = to_lower_str(current);
            if (lower.size() >= 2 && STOPWORDS.find(lower) == STOPWORDS.end()) {
                tokens.push_back(current);
            }
            current.clear();
        }
    }
    if (!current.empty()) {
        std::string lower = to_lower_str(current);
        if (lower.size() >= 2 && STOPWORDS.find(lower) == STOPWORDS.end()) {
            tokens.push_back(current);
        }
    }

    return tokens;
}

} // namespace

uint64_t CanonicalHistoryStore::append_turn(const std::string& role, const std::string& content, int32_t token_start, int32_t token_end) {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t id = next_turn_id_++;
    HistoryTurn turn;
    turn.turn_id = id;
    turn.role = role;
    turn.content = content;
    turn.token_start = token_start;
    turn.token_end = token_end;
    turn.is_evicted_from_kv = false;
    turns_.push_back(std::move(turn));
    return id;
}

void CanonicalHistoryStore::mark_evicted_range(int32_t evict_start, int32_t evict_end) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& turn : turns_) {
        // If turn token range overlaps with evicted sequence interval [evict_start, evict_end)
        if (turn.token_start < evict_end && turn.token_end > evict_start) {
            turn.is_evicted_from_kv = true;
        }
    }
}

const std::vector<HistoryTurn>& CanonicalHistoryStore::get_turns() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return turns_;
}

const HistoryTurn* CanonicalHistoryStore::get_turn(uint64_t turn_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& t : turns_) {
        if (t.turn_id == turn_id) return &t;
    }
    return nullptr;
}

const HistoryTurn* CanonicalHistoryStore::find_turn_by_token(int32_t token_pos) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& t : turns_) {
        if (token_pos >= t.token_start && token_pos < t.token_end) {
            return &t;
        }
    }
    return nullptr;
}

std::vector<std::pair<uint64_t, float>> CanonicalHistoryStore::search_lexical(const std::string& query_text, bool evicted_only) const {
    std::vector<std::string> query_words = extract_query_keywords(query_text);
    if (query_words.empty()) {
        return {};
    }

    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::pair<uint64_t, float>> matches;

    for (const auto& turn : turns_) {
        if (evicted_only && !turn.is_evicted_from_kv) {
            continue;
        }

        std::string lower_content = to_lower_str(turn.content);
        float raw_score = 0.0f;
        float max_possible = 0.0f;

        for (const auto& qw : query_words) {
            std::string lower_qw = to_lower_str(qw);
            bool contains_digit = std::any_of(qw.begin(), qw.end(), [](char c) { return std::isdigit(c); });
            bool contains_special = (qw.find('-') != std::string::npos || qw.find('_') != std::string::npos);
            float weight = (contains_digit || contains_special) ? 4.0f : 1.5f;

            max_possible += weight;

            // Direct substring search
            if (lower_content.find(lower_qw) != std::string::npos) {
                raw_score += weight;
            }
        }

        if (raw_score > 0.0f && max_possible > 0.0f) {
            float normalized = raw_score / max_possible;
            matches.push_back({turn.turn_id, normalized});
        }
    }

    std::sort(matches.begin(), matches.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    return matches;
}

std::string CanonicalHistoryStore::format_system_memory_context(const std::vector<RetrievedMemory>& memories, size_t max_chars) const {
    if (memories.empty()) {
        return "";
    }

    std::ostringstream oss;
    oss << "[Memori Percakapan Terkait (Riwayat yang Diarsipkan)]:\n";

    size_t chars_written = 0;
    for (const auto& m : memories) {
        std::ostringstream turn_stream;
        turn_stream << "- [Turn " << m.turn_id << "] " << m.role << ": " << m.content << "\n";
        std::string line = turn_stream.str();

        if (chars_written + line.size() > max_chars) {
            break;
        }
        oss << line;
        chars_written += line.size();
    }

    oss << "Instruksi Penting: Gunakan fakta historis di atas untuk menjawab jika relevan. Jangan mengarang fakta.\n";
    return oss.str();
}

void CanonicalHistoryStore::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    turns_.clear();
    next_turn_id_ = 0;
}

size_t CanonicalHistoryStore::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return turns_.size();
}

} // namespace amuquant
