#include "amuquant/canonical_history.hpp"

#include <algorithm>
#include <sstream>
#include <cctype>
#include <unordered_set>
#include <cmath>
#include <regex>

namespace amuquant {

namespace {

std::string to_lower_str(const std::string& str) {
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    return lower;
}

std::string trim_string(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n\"'");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n\"'.,;");
    return str.substr(first, (last - first + 1));
}

bool is_alphanumeric_or_symbol(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_';
}

std::vector<std::string> extract_query_keywords(const std::string& text) {
    static const std::unordered_set<std::string> STOPWORDS = {
        "dan", "di", "ke", "dari", "yang", "ini", "itu", "untuk", "pada", "adalah",
        "apa", "siapa", "berapa", "dimana", "kapan", "bagaimana", "mengapa", "apakah",
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for",
        "of", "with", "by", "what", "who", "which", "where", "how", "many"
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

struct ParsedFact {
    std::string key;
    std::string value;
    std::string raw;
};

std::vector<ParsedFact> extract_facts_from_content(const std::string& content) {
    std::vector<ParsedFact> extracted;

    // Split into sentences / segments by newline, semicolon, comma, or period
    std::vector<std::string> segments;
    std::string current;
    for (size_t i = 0; i < content.size(); ++i) {
        char c = content[i];
        bool is_period_delim = (c == '.' && (i + 1 == content.size() || std::isspace(static_cast<unsigned char>(content[i+1]))));
        if (c == '\n' || c == ';' || c == ',' || is_period_delim) {
            std::string trimmed = trim_string(current);
            if (!trimmed.empty()) segments.push_back(trimmed);
            current.clear();
        } else {
            current.push_back(c);
        }
    }
    std::string trimmed_tail = trim_string(current);
    if (!trimmed_tail.empty()) segments.push_back(trimmed_tail);

    for (auto seg : segments) {
        // Strip common prefix headers (e.g., "Pengumuman:", "Catatan:", "Update:", "Note:", "Info:")
        static const std::vector<std::string> HEADER_PREFIXES = {
            "pengumuman:", "catatan:", "update:", "pemberitahuan:", "note:", "notice:", "info:"
        };
        std::string lower_initial = to_lower_str(seg);
        for (const auto& hp : HEADER_PREFIXES) {
            if (lower_initial.rfind(hp, 0) == 0) {
                seg = trim_string(seg.substr(hp.size()));
                break;
            }
        }
        if (seg.empty()) continue;

        std::string lower_seg = to_lower_str(seg);

        // Pattern 1: Verbal update/assertion markers: "X diubah menjadi Y", "X updated to Y", "X adalah Y"
        static const std::vector<std::string> VERBAL_MARKERS = {
            " diubah menjadi ", " updated to ", " changed to ", " berganti menjadi ",
            " adalah ", " is ", " was ", " diset ke ", " set to ", " bernilai "
        };

        bool matched_verbal = false;
        for (const auto& marker : VERBAL_MARKERS) {
            size_t mpos = lower_seg.find(marker);
            if (mpos != std::string::npos && mpos > 0) {
                std::string k = trim_string(seg.substr(0, mpos));
                std::string v = trim_string(seg.substr(mpos + marker.size()));
                if (!k.empty() && !v.empty() && k.size() <= 40 && v.size() <= 120) {
                    extracted.push_back({k, v, seg});
                    matched_verbal = true;
                    break;
                }
            }
        }
        if (matched_verbal) continue;

        // Pattern 2: key : value or key = value
        size_t delim_pos = seg.find(':');
        if (delim_pos == std::string::npos) {
            delim_pos = seg.find('=');
        }
        if (delim_pos != std::string::npos && delim_pos > 0 && delim_pos + 1 < seg.size()) {
            std::string k = trim_string(seg.substr(0, delim_pos));
            std::string v = trim_string(seg.substr(delim_pos + 1));
            if (!k.empty() && !v.empty() && k.size() <= 40 && v.size() <= 120) {
                extracted.push_back({k, v, seg});
                continue;
            }
        }

        // Pattern 3: Common Indonesian entity naming phrases: "nama proyeknya X", "nomor versinya X", "pemiliknya X"
        static const std::vector<std::pair<std::string, std::string>> PHRASE_PATTERNS = {
            {"nama proyeknya ", "nama proyek"},
            {"nama proyek ", "nama proyek"},
            {"nomor versinya ", "nomor versi"},
            {"nomor versi ", "nomor versi"},
            {"pemiliknya ", "pemilik"},
            {"pemilik ", "pemilik"},
            {"kode rahasianya ", "kode rahasia"},
            {"kode rahasia ", "kode rahasia"},
            {"project name is ", "project name"},
            {"version number is ", "version number"},
            {"secret code is ", "secret code"}
        };

        for (const auto& pp : PHRASE_PATTERNS) {
            size_t ppos = lower_seg.find(pp.first);
            if (ppos != std::string::npos) {
                std::string v = trim_string(seg.substr(ppos + pp.first.size()));
                // If value has trailing conjunctions like "dan...", clean it up
                size_t and_pos = to_lower_str(v).find(" dan ");
                if (and_pos != std::string::npos) {
                    v = trim_string(v.substr(0, and_pos));
                }
                if (!v.empty() && v.size() <= 80) {
                    extracted.push_back({pp.second, v, seg});
                    break;
                }
            }
        }
    }

    return extracted;
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

    // Automatically extract factual statements and track conflicts
    auto parsed = extract_facts_from_content(content);
    for (const auto& pf : parsed) {
        std::string clean_key = trim_string(to_lower_str(pf.key));
        std::string clean_val = trim_string(pf.value);
        if (clean_key.empty() || clean_val.empty()) continue;

        // Check for temporal conflict / superseding
        for (auto& existing : facts_) {
            if (existing.status == FactStatus::ACTIVE) {
                std::string ex_key = trim_string(to_lower_str(existing.entity_key));
                if (ex_key == clean_key) {
                    std::string ex_val = trim_string(to_lower_str(existing.entity_value));
                    std::string cur_val = trim_string(to_lower_str(clean_val));
                    if (ex_val != cur_val) {
                        existing.status = FactStatus::SUPERSEDED;
                        existing.superseded_by_turn = id;
                        conflict_count_++;
                    }
                }
            }
        }

        FactSlot slot;
        slot.slot_id = next_slot_id_++;
        slot.turn_id = id;
        slot.role = role;
        slot.entity_key = pf.key;
        slot.entity_value = clean_val;
        slot.status = FactStatus::ACTIVE;
        slot.superseded_by_turn = 0;
        slot.raw_statement = pf.raw;
        facts_.push_back(std::move(slot));
    }

    return id;
}

void CanonicalHistoryStore::mark_evicted_range(int32_t evict_start, int32_t evict_end) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& turn : turns_) {
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

uint64_t CanonicalHistoryStore::register_fact(
    uint64_t turn_id,
    const std::string& role,
    const std::string& entity_key,
    const std::string& entity_value,
    const std::string& raw_statement
) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string clean_key = trim_string(to_lower_str(entity_key));
    std::string clean_val = trim_string(entity_value);
    if (clean_key.empty() || clean_val.empty()) {
        return 0;
    }

    // Check for temporal conflict / superseding
    for (auto& existing : facts_) {
        if (existing.status == FactStatus::ACTIVE) {
            std::string ex_key = trim_string(to_lower_str(existing.entity_key));
            if (ex_key == clean_key) {
                std::string ex_val = trim_string(to_lower_str(existing.entity_value));
                std::string cur_val = trim_string(to_lower_str(clean_val));
                if (ex_val != cur_val) {
                    existing.status = FactStatus::SUPERSEDED;
                    existing.superseded_by_turn = turn_id;
                    conflict_count_++;
                } else {
                    return existing.slot_id;
                }
            }
        }
    }

    uint64_t slot_id = next_slot_id_++;
    FactSlot slot;
    slot.slot_id = slot_id;
    slot.turn_id = turn_id;
    slot.role = role;
    slot.entity_key = entity_key;
    slot.entity_value = clean_val;
    slot.status = FactStatus::ACTIVE;
    slot.superseded_by_turn = 0;
    slot.raw_statement = raw_statement.empty() ? (entity_key + ": " + clean_val) : raw_statement;
    facts_.push_back(std::move(slot));
    return slot_id;
}

void CanonicalHistoryStore::extract_and_register_facts(uint64_t turn_id, const std::string& role, const std::string& content) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto parsed = extract_facts_from_content(content);
    for (const auto& pf : parsed) {
        std::string clean_key = trim_string(to_lower_str(pf.key));
        std::string clean_val = trim_string(pf.value);
        if (clean_key.empty() || clean_val.empty()) continue;

        for (auto& existing : facts_) {
            if (existing.status == FactStatus::ACTIVE) {
                std::string ex_key = trim_string(to_lower_str(existing.entity_key));
                if (ex_key == clean_key) {
                    std::string ex_val = trim_string(to_lower_str(existing.entity_value));
                    std::string cur_val = trim_string(to_lower_str(clean_val));
                    if (ex_val != cur_val) {
                        existing.status = FactStatus::SUPERSEDED;
                        existing.superseded_by_turn = turn_id;
                        conflict_count_++;
                    }
                }
            }
        }

        FactSlot slot;
        slot.slot_id = next_slot_id_++;
        slot.turn_id = turn_id;
        slot.role = role;
        slot.entity_key = pf.key;
        slot.entity_value = clean_val;
        slot.status = FactStatus::ACTIVE;
        slot.superseded_by_turn = 0;
        slot.raw_statement = pf.raw;
        facts_.push_back(std::move(slot));
    }
}

std::vector<FactSlot> CanonicalHistoryStore::get_facts(bool active_only) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!active_only) return facts_;

    std::vector<FactSlot> active;
    for (const auto& f : facts_) {
        if (f.status == FactStatus::ACTIVE) active.push_back(f);
    }
    return active;
}

const FactSlot* CanonicalHistoryStore::find_fact_by_key(const std::string& entity_key, bool active_only) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string lower_k = trim_string(to_lower_str(entity_key));
    for (const auto& f : facts_) {
        if (active_only && f.status != FactStatus::ACTIVE) continue;
        if (trim_string(to_lower_str(f.entity_key)) == lower_k) {
            return &f;
        }
    }
    return nullptr;
}

std::vector<FactSlot> CanonicalHistoryStore::search_facts(const std::string& query_text, bool active_only) const {
    std::vector<std::string> query_words = extract_query_keywords(query_text);
    if (query_words.empty()) return {};

    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<FactSlot> results;

    for (const auto& f : facts_) {
        if (active_only && f.status != FactStatus::ACTIVE) continue;
        std::string lower_k = to_lower_str(f.entity_key);
        std::string lower_v = to_lower_str(f.entity_value);
        for (const auto& qw : query_words) {
            std::string lower_qw = to_lower_str(qw);
            if (lower_k.find(lower_qw) != std::string::npos || lower_v.find(lower_qw) != std::string::npos) {
                results.push_back(f);
                break;
            }
        }
    }

    return results;
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

    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream oss;
    oss << "[RELIABLE CONVERSATIONAL MEMORY (Historical Provenance)]:\n";

    size_t chars_written = 0;
    for (const auto& m : memories) {
        std::ostringstream turn_stream;
        turn_stream << "- [Turn " << m.turn_id << " | Source: " << m.role << " | Status: ACTIVE] " << m.content << "\n";
        std::string line = turn_stream.str();

        if (chars_written + line.size() > max_chars) {
            break;
        }
        oss << line;
        chars_written += line.size();
    }

    oss << "Instruksi: Gunakan fakta terverifikasi di atas. Sitasi nomor [Turn X] saat merujuk informasi masa lalu dan jangan mengarang fakta.\n";
    return oss.str();
}

std::string CanonicalHistoryStore::format_grounded_memory_prompt(
    const std::string& query_text,
    float evidence_threshold,
    float* out_max_score,
    bool* out_abstain_recommended,
    size_t max_chars
) const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<std::string> query_words = extract_query_keywords(query_text);
    float max_score = 0.0f;

    std::vector<FactSlot> matched_facts;
    for (const auto& f : facts_) {
        if (f.status != FactStatus::ACTIVE) continue;
        std::vector<std::string> k_words = extract_query_keywords(f.entity_key);
        if (k_words.empty()) continue;

        size_t overlap = 0;
        for (const auto& kw : k_words) {
            std::string lower_kw = to_lower_str(kw);
            for (const auto& qw : query_words) {
                if (to_lower_str(qw) == lower_kw) {
                    overlap++;
                    break;
                }
            }
        }

        float overlap_ratio = static_cast<float>(overlap) / static_cast<float>(k_words.size());
        bool key_matched = (k_words.size() == 1) ? (overlap == 1) : (overlap_ratio >= 0.60f);

        bool val_matched = false;
        std::string lower_v = to_lower_str(f.entity_value);
        for (const auto& qw : query_words) {
            if (to_lower_str(qw) == lower_v) {
                val_matched = true;
                break;
            }
        }

        if (key_matched || val_matched) {
            matched_facts.push_back(f);
            max_score = std::max(max_score, 0.90f);
        }
    }

    // Lexical check across dialogue turns
    for (const auto& t : turns_) {
        std::string lower_c = to_lower_str(t.content);
        float turn_score = 0.0f;
        float max_possible = 0.0f;
        for (const auto& qw : query_words) {
            std::string lower_qw = to_lower_str(qw);
            float weight = 1.0f;
            max_possible += weight;
            if (lower_c.find(lower_qw) != std::string::npos) {
                turn_score += weight;
            }
        }
        if (max_possible > 0.0f) {
            max_score = std::max(max_score, turn_score / max_possible);
        }
    }

    if (out_max_score) *out_max_score = max_score;

    bool abstain = (max_score < evidence_threshold || (query_words.size() > 0 && matched_facts.empty() && max_score < 0.40f));
    if (out_abstain_recommended) *out_abstain_recommended = abstain;

    std::ostringstream oss;
    if (abstain) {
        oss << "[RELIABLE MEMORY GUARD: INSUFFICIENT EVIDENCE]\n";
        oss << "Tidak ditemukan catatan fakta dalam riwayat percakapan mengenai pertanyaan: \"" << query_text << "\".\n";
        oss << "Instruksi Wajib:\n";
        oss << "1. Anda WAJIB menjawab secara eksplisit bahwa informasi tersebut belum tercatat atau tidak diketahui dalam riwayat percakapan.\n";
        oss << "2. DILARANG keras menebak, berspekulasi, atau mengarang fakta yang tidak tercatat.\n";
        return oss.str();
    }

    oss << "[RELIABLE CONVERSATIONAL MEMORY (Fakta Riwayat Terverifikasi)]\n";
    oss << "Daftar fakta aktif terverifikasi beserta nomor turn sumbernya:\n";
    size_t chars_written = 0;
    for (const auto& mf : matched_facts) {
        std::ostringstream item;
        item << "- [Turn " << mf.turn_id << " | Sumber: " << mf.role << " | Status: ACTIVE] "
             << mf.entity_key << ": " << mf.entity_value << "\n";
        std::string s = item.str();
        if (chars_written + s.size() > max_chars) break;
        oss << s;
        chars_written += s.size();
    }

    // Append notes for superseded facts of matching entities
    for (const auto& f : facts_) {
        if (f.status == FactStatus::SUPERSEDED) {
            for (const auto& mf : matched_facts) {
                if (to_lower_str(f.entity_key) == to_lower_str(mf.entity_key)) {
                    std::ostringstream item;
                    item << "  (Catatan: Menggantikan / Supersedes Turn " << f.turn_id << " " << f.entity_key << ": " << f.entity_value << ")\n";
                    std::string s = item.str();
                    if (chars_written + s.size() <= max_chars) {
                        oss << s;
                        chars_written += s.size();
                    }
                }
            }
        }
    }

    oss << "Instruksi:\n";
    oss << "1. Jawab pertanyaan hanya berdasarkan fakta ACTIVE di atas.\n";
    oss << "2. Sebutkan nilai fakta dan sitasi nomor [Turn X] sumbernya.\n";
    oss << "3. Jangan gunakan nilai yang telah digantikan (Superseded).\n";

    return oss.str();
}

void CanonicalHistoryStore::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    turns_.clear();
    facts_.clear();
    next_turn_id_ = 0;
    next_slot_id_ = 0;
    conflict_count_ = 0;
}

size_t CanonicalHistoryStore::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return turns_.size();
}

} // namespace amuquant
