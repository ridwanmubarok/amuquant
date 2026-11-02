/**
 * @file test_reliable_memory.cpp
 * @brief Unit tests for AMU-QUANT Reliable Memory: Provenance tracking, FactRegister,
 *        Temporal Conflict Resolution, and Calibrated Abstention.
 */

#include "amuquant/canonical_history.hpp"
#include "amuquant/engine.hpp"

#include <cassert>
#include <iostream>
#include <string>

using namespace amuquant;

void test_provenance_tracking() {
    std::cout << "[TEST 1] Verifying Provenance Tracking and Fact Extraction..." << std::endl;
    CanonicalHistoryStore history;

    uint64_t t0 = history.append_turn("user", "Halo, nama saya Budi dan saya bekerja di sistem Aegis.", 0, 15);
    uint64_t t1 = history.append_turn("assistant", "Halo Budi! Ada yang bisa saya bantu terkait Aegis?", 16, 30);
    uint64_t t2 = history.append_turn("user", "Port server: 8080. Kode rahasia: ALPHA-99.", 31, 45);

    assert(t0 == 0);
    assert(t1 == 1);
    assert(t2 == 2);
    assert(history.size() == 3);

    auto facts = history.get_facts(true);
    assert(facts.size() >= 2);

    bool found_port = false;
    bool found_code = false;
    for (const auto& f : facts) {
        if (f.entity_key == "Port server" && f.entity_value == "8080") {
            assert(f.turn_id == 2);
            assert(f.role == "user");
            assert(f.status == FactStatus::ACTIVE);
            found_port = true;
        }
        if (f.entity_key == "Kode rahasia" && f.entity_value == "ALPHA-99") {
            assert(f.turn_id == 2);
            assert(f.role == "user");
            assert(f.status == FactStatus::ACTIVE);
            found_code = true;
        }
    }
    assert(found_port);
    assert(found_code);

    std::cout << "  Passed: Provenance tags and facts extracted cleanly." << std::endl;
}

void test_temporal_conflict_resolution() {
    std::cout << "[TEST 2] Verifying Temporal Fact Conflict Resolution..." << std::endl;
    CanonicalHistoryStore history;

    // Turn 0: initial fact
    history.append_turn("user", "Nama proyek: Aegis-V1. Status: Development.", 0, 15);
    auto facts_init = history.get_facts(true);
    assert(facts_init.size() == 2);
    assert(history.get_conflict_count() == 0);

    // Turn 1: filler
    history.append_turn("assistant", "Baik, dicatat.", 16, 20);

    // Turn 2: fact update / conflict introduction
    history.append_turn("user", "Pengumuman: Nama proyek diubah menjadi Aegis-NextGen. Status: Production.", 21, 40);

    // Verify conflict tracking
    assert(history.get_conflict_count() == 2); // Both 'nama proyek' and 'status' updated

    auto active_facts = history.get_facts(true);
    assert(active_facts.size() == 2);

    for (const auto& f : active_facts) {
        if (f.entity_key == "nama proyek" || f.entity_key == "Nama proyek") {
            assert(f.entity_value == "Aegis-NextGen");
            assert(f.status == FactStatus::ACTIVE);
            assert(f.turn_id == 2);
        }
        if (f.entity_key == "Status" || f.entity_key == "status") {
            assert(f.entity_value == "Production");
            assert(f.status == FactStatus::ACTIVE);
            assert(f.turn_id == 2);
        }
    }

    // Verify all facts (including superseded)
    auto all_facts = history.get_facts(false);
    assert(all_facts.size() == 4);
    int superseded_count = 0;
    for (const auto& f : all_facts) {
        if (f.status == FactStatus::SUPERSEDED) {
            superseded_count++;
            assert(f.superseded_by_turn == 2);
        }
    }
    assert(superseded_count == 2);

    // Grounded prompt formatting check
    float max_score = 0.0f;
    bool abstain = false;
    std::string prompt = history.format_grounded_memory_prompt("Berapa status proyek sekarang?", 0.25f, &max_score, &abstain);
    assert(!abstain);
    assert(max_score >= 0.70f);
    assert(prompt.find("Production") != std::string::npos);
    assert(prompt.find("Supersedes Turn 0") != std::string::npos);

    std::cout << "  Passed: Temporal conflicts resolved and superseded records tagged." << std::endl;
}

void test_calibrated_abstention() {
    std::cout << "[TEST 3] Verifying Calibrated Abstention Guardrail..." << std::endl;
    AMUQuantConfig cfg;
    cfg.evidence_threshold = 0.35f;
    cfg.enable_abstention_guardrail = true;

    AMUQuantEngine engine(cfg);
    engine.record_turn("user", "Nama proyek: Aegis-V1. Server: Ubuntu 24.04.", 0, 15);
    engine.record_turn("assistant", "Sistem siap.", 16, 20);

    // Query with non-existent factual request
    float rel_score = 0.0f;
    bool abstain_rec = false;
    std::string guard_prompt = engine.construct_grounded_prompt("Berapa nomor lisensi rahasia bank?", &rel_score, &abstain_rec);

    assert(abstain_rec == true);
    assert(rel_score < cfg.evidence_threshold);
    assert(guard_prompt.find("[RELIABLE MEMORY GUARD: INSUFFICIENT EVIDENCE]") != std::string::npos);

    // Verify response grounding check
    std::string compliant_response = "Maaf, informasi tentang nomor lisensi rahasia bank tidak tercatat dalam riwayat percakapan.";
    bool check1 = engine.verify_response_grounding(compliant_response, guard_prompt);
    assert(check1 == true);

    std::string hallucinating_response = "Nomor lisensi rahasia bank adalah BANK-99412-SEC.";
    bool check2 = engine.verify_response_grounding(hallucinating_response, guard_prompt);
    assert(check2 == false);

    auto m = engine.get_reliable_memory_metrics();
    assert(m.abstentions_recommended == 1);
    assert(m.grounding_verifications_passed == 1);
    assert(m.grounding_verifications_failed == 1);

    std::cout << "  Passed: Calibrated abstention guardrail accurately triggered and verified." << std::endl;
}

void test_prompt_injection_after_eviction() {
    std::cout << "[TEST 4] Verifying Fact Retention and Prompt Grounding Post-Eviction..." << std::endl;
    AMUQuantConfig cfg;
    cfg.evidence_threshold = 0.25f;
    cfg.enable_abstention_guardrail = true;

    AMUQuantEngine engine(cfg);
    Status st = engine.initialize();
    assert(st == Status::OK);

    // Turn 0: User introduces factual parameters
    uint64_t t0 = engine.record_turn("user", "Host server: 192.168.1.150. Port: 9000. Service: Aegis-Gateway.", 0, 20);
    assert(t0 == 0);

    // Turn 1: Assistant acknowledges
    engine.record_turn("assistant", "Konfigurasi server 192.168.1.150 pada port 9000 telah aktif.", 21, 35);

    // Verify facts were extracted into FactRegister
    auto facts_before = engine.get_history_store()->get_facts(true);
    assert(facts_before.size() >= 3);

    // Simulate physical KV eviction of Turn 0 & Turn 1 tokens [0, 36)
    engine.get_history_store()->mark_evicted_range(0, 36);

    // Verify that the turns are marked as evicted from active KV-cache
    const auto& turns = engine.get_history_store()->get_turns();
    assert(turns.size() == 2);
    assert(turns[0].is_evicted_from_kv == true);
    assert(turns[1].is_evicted_from_kv == true);

    // Turn 2: User queries evicted factual information: "Berapa port dan host server?"
    float rel = 0.0f;
    bool abstain = false;
    std::string prompt_block = engine.construct_grounded_prompt("Berapa port dan host server?", &rel, &abstain);

    // Fact must be recalled from canonical history despite physical KV-cache eviction!
    assert(!abstain);
    assert(rel >= cfg.evidence_threshold);
    assert(prompt_block.find("192.168.1.150") != std::string::npos);
    assert(prompt_block.find("9000") != std::string::npos);
    assert(prompt_block.find("[RELIABLE CONVERSATIONAL MEMORY") != std::string::npos);

    // Verify grounding check for assistant answer
    std::string accurate_answer = "Port server adalah 9000 dan host server adalah 192.168.1.150.";
    assert(engine.verify_response_grounding(accurate_answer, prompt_block) == true);

    std::cout << "  Passed: Evicted factual context successfully recalled and reconstructed." << std::endl;
}

void test_intent_aware_abstention_guardrail() {
    std::cout << "[TEST 5] Verifying Intent-Aware Calibrated Abstention (No Over-Refusal)..." << std::endl;
    AMUQuantConfig cfg;
    cfg.enable_abstention_guardrail = true;

    AMUQuantEngine engine(cfg);
    engine.initialize();

    // 1. General conversation greeting: must NOT trigger abstention guard
    float rel1 = 0.0f;
    bool abstain1 = true;
    std::string prompt1 = engine.construct_grounded_prompt("Halo, selamat pagi! Bagaimana kabarmu?", &rel1, &abstain1);
    assert(!abstain1);
    assert(prompt1.empty());

    // 2. General programming request: must NOT trigger abstention guard
    float rel2 = 0.0f;
    bool abstain2 = true;
    std::string prompt2 = engine.construct_grounded_prompt("Tolong buatkan fungsi binary search di C++.", &rel2, &abstain2);
    assert(!abstain2);
    assert(prompt2.empty());

    // 3. Inquiry regarding past factual parameter not in memory: MUST trigger abstention guard
    float rel3 = 0.0f;
    bool abstain3 = false;
    std::string prompt3 = engine.construct_grounded_prompt("Berapa kode rahasia vault yang tadi disebutkan?", &rel3, &abstain3);
    assert(abstain3);
    assert(!prompt3.empty());
    assert(prompt3.find("[RELIABLE MEMORY GUARD: INSUFFICIENT EVIDENCE]") != std::string::npos);

    std::cout << "  Passed: Intent-aware abstention distinguishes chat from factual inquiries." << std::endl;
}

void test_rolling_window_compaction() {
    std::cout << "[TEST 6] Verifying Rolling Window Canonical History Compaction (O(1) Bound)..." << std::endl;
    CanonicalHistoryStore store;
    store.set_max_retained_turns(10);

    // Turn 0 has an active fact
    store.append_turn("user", "Host server: 10.0.0.1.", 0, 5);

    // Add 20 filler turns without facts
    for (int i = 1; i <= 20; ++i) {
        store.append_turn("user", "Percakapan biasa filler nomor " + std::to_string(i), i * 10, i * 10 + 5);
    }

    // Size must be bounded around max_retained_turns_ (10)
    assert(store.size() <= 11);

    // Turn 0 with active fact must be preserved despite being oldest!
    auto facts = store.get_facts(true);
    assert(!facts.empty());
    assert(facts[0].entity_key == "Host server");
    assert(facts[0].entity_value == "10.0.0.1");

    std::cout << "  Passed: Rolling window bounded turns without losing active factual slots." << std::endl;
}

int main() {
    std::cout << "=== AMU-QUANT RELIABLE MEMORY UNIT TEST SUITE ===" << std::endl;
    test_provenance_tracking();
    test_temporal_conflict_resolution();
    test_calibrated_abstention();
    test_prompt_injection_after_eviction();
    test_intent_aware_abstention_guardrail();
    test_rolling_window_compaction();
    std::cout << "ALL RELIABLE MEMORY TESTS PASSED SUCCESSFULLY!" << std::endl;
    return 0;
}
