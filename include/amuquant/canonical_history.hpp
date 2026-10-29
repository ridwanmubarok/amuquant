#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <mutex>
#include <utility>

namespace amuquant {

enum class FactStatus {
    ACTIVE,
    SUPERSEDED,
    CONTRADICTED
};

struct FactSlot {
    uint64_t slot_id{0};
    uint64_t turn_id{0};
    std::string role;
    std::string entity_key;
    std::string entity_value;
    FactStatus status{FactStatus::ACTIVE};
    uint64_t superseded_by_turn{0};
    std::string raw_statement;
};

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

    // Fact Register and Provenance Tracking
    uint64_t register_fact(
        uint64_t turn_id,
        const std::string& role,
        const std::string& entity_key,
        const std::string& entity_value,
        const std::string& raw_statement = ""
    );
    void extract_and_register_facts(uint64_t turn_id, const std::string& role, const std::string& content);
    std::vector<FactSlot> get_facts(bool active_only = false) const;
    const FactSlot* find_fact_by_key(const std::string& entity_key, bool active_only = true) const;
    std::vector<FactSlot> search_facts(const std::string& query_text, bool active_only = true) const;
    uint32_t get_conflict_count() const { return conflict_count_; }

    // Lexical entity and keyword matching across turns
    std::vector<std::pair<uint64_t, float>> search_lexical(const std::string& query_text, bool evicted_only = false) const;

    // Formats retrieved historical memories into a structured system prompt block with provenance tags
    std::string format_system_memory_context(const std::vector<RetrievedMemory>& memories, size_t max_chars = 1500) const;

    // Constructs grounded memory prompt with provenance tags and calibrated abstention
    std::string format_grounded_memory_prompt(
        const std::string& query_text,
        float evidence_threshold = 0.25f,
        float* out_max_score = nullptr,
        bool* out_abstain_recommended = nullptr,
        size_t max_chars = 1500
    ) const;

    void clear();
    size_t size() const;

private:
    mutable std::mutex mutex_;
    std::vector<HistoryTurn> turns_;
    std::vector<FactSlot> facts_;
    uint64_t next_turn_id_{0};
    uint64_t next_slot_id_{0};
    uint32_t conflict_count_{0};
};

} // namespace amuquant
