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

int main() {
    std::cout << "=== AMU-QUANT RELIABLE MEMORY UNIT TEST SUITE ===" << std::endl;
    test_provenance_tracking();
    test_temporal_conflict_resolution();
    test_calibrated_abstention();
    std::cout << "ALL RELIABLE MEMORY TESTS PASSED SUCCESSFULLY!" << std::endl;
    return 0;
}
