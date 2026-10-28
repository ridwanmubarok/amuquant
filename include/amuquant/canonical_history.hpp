#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <mutex>
#include <utility>

namespace amuquant {

struct HistoryTurn {
    uint64_t turn_id{0};
    std::string role;        // "user", "assistant", "system", "tool"
    std::string content;     // Raw verbatim text
    int32_t token_start{0};
    int32_t token_end{0};
    bool is_evicted_from_kv{false}; // True if physical attention slots were pruned from llama.cpp
};

struct RetrievedMemory {
    uint64_t turn_id{0};
    std::string role;
    std::string content;
    float dense_score{0.0f};
    float lexical_score{0.0f};
    float combined_score{0.0f};
};

class CanonicalHistoryStore {
public:
    CanonicalHistoryStore() = default;

    uint64_t append_turn(const std::string& role, const std::string& content, int32_t token_start, int32_t token_end);
    void mark_evicted_range(int32_t evict_start, int32_t evict_end);

    const std::vector<HistoryTurn>& get_turns() const;
    const HistoryTurn* get_turn(uint64_t turn_id) const;
    const HistoryTurn* find_turn_by_token(int32_t token_pos) const;

    // Lexical entity and keyword matching across turns
    std::vector<std::pair<uint64_t, float>> search_lexical(const std::string& query_text, bool evicted_only = false) const;

    // Formats retrieved historical memories into a structured system prompt block
    std::string format_system_memory_context(const std::vector<RetrievedMemory>& memories, size_t max_chars = 1500) const;

    void clear();
    size_t size() const;

private:
    mutable std::mutex mutex_;
    std::vector<HistoryTurn> turns_;
    uint64_t next_turn_id_{0};
};

} // namespace amuquant
