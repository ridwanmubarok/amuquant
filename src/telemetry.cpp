#include "amuquant/telemetry.hpp"

#include <chrono>
#include <sys/resource.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/task_info.h>
#include <mach/mach_host.h>
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <unistd.h>
#include <fstream>
#include <sstream>
#endif

namespace amuquant {

SystemTelemetry::SystemTelemetry() = default;

double SystemTelemetry::get_time_seconds() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now.time_since_epoch()).count();
}

void SystemTelemetry::set_mock_mode(bool enable) {
    mock_mode_ = enable;
}

void SystemTelemetry::set_mock_stats(const MemoryStats& stats) {
    mock_stats_ = stats;
}

MemoryStats SystemTelemetry::sample() {
    if (mock_mode_) {
        if (mock_stats_.timestamp_sec <= 0.0) {
            mock_stats_.timestamp_sec = get_time_seconds();
        }
        return mock_stats_;
    }

    MemoryStats stats{};
    stats.timestamp_sec = get_time_seconds();

    // 1. Ambil data Major Page Faults via getrusage
    struct rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        stats.major_page_faults = static_cast<uint64_t>(usage.ru_majflt);
    }

#if defined(__APPLE__)
    // 2. Ambil RSS proses pada macOS via task_info (phys_footprint)
    task_vm_info_data_t vm_info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    kern_return_t kr = task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&vm_info), &count);
    if (kr == KERN_SUCCESS) {
        stats.process_rss_bytes = static_cast<uint64_t>(vm_info.phys_footprint);
    } else {
        // Fallback ke ru_maxrss
        stats.process_rss_bytes = static_cast<uint64_t>(usage.ru_maxrss);
    }

    // 3. Ambil total RAM fisik sistem
    int mib[2] = {CTL_HW, HW_MEMSIZE};
    uint64_t total_ram = 0;
    size_t length = sizeof(total_ram);
    if (sysctl(mib, 2, &total_ram, &length, nullptr, 0) == 0) {
        stats.total_ram_bytes = total_ram;
    }

    // 4. Ambil RAM yang tersedia (Free + Inactive + Speculative)
    mach_port_t host_port = mach_host_self();
    vm_size_t page_size = 0;
    host_page_size(host_port, &page_size);

    vm_statistics64_data_t host_vm_info{};
    mach_msg_type_number_t host_count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(host_port, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&host_vm_info), &host_count) == KERN_SUCCESS) {
        uint64_t free_bytes = static_cast<uint64_t>(host_vm_info.free_count) * page_size;
        uint64_t inactive_bytes = static_cast<uint64_t>(host_vm_info.inactive_count) * page_size;
        uint64_t speculative_bytes = static_cast<uint64_t>(host_vm_info.speculative_count) * page_size;
        stats.available_ram_bytes = free_bytes + inactive_bytes + speculative_bytes;
    }

    // 5. Ambil swap usage
    struct xsw_usage swap_usage{};
    size_t swap_len = sizeof(swap_usage);
    if (sysctlbyname("vm.swapusage", &swap_usage, &swap_len, nullptr, 0) == 0) {
        stats.swap_used_bytes = static_cast<uint64_t>(swap_usage.xsu_used);
    }

#elif defined(__linux__)
    // Implementasi Linux via /proc/meminfo dan /proc/self/statm
    long page_size = sysconf(_SC_PAGESIZE);

    std::ifstream statm("/proc/self/statm");
    if (statm.is_open()) {
        uint64_t size_pages = 0;
        uint64_t resident_pages = 0;
        statm >> size_pages >> resident_pages;
        stats.process_rss_bytes = resident_pages * page_size;
    }

    std::ifstream meminfo("/proc/meminfo");
    if (meminfo.is_open()) {
        std::string line;
        uint64_t swap_total = 0;
        uint64_t swap_free = 0;
        while (std::getline(meminfo, line)) {
            std::istringstream iss(line);
            std::string key;
            uint64_t val_kb = 0;
            if (iss >> key >> val_kb) {
                if (key == "MemTotal:") {
                    stats.total_ram_bytes = val_kb * 1024ULL;
                } else if (key == "MemAvailable:") {
                    stats.available_ram_bytes = val_kb * 1024ULL;
                } else if (key == "SwapTotal:") {
                    swap_total = val_kb * 1024ULL;
                } else if (key == "SwapFree:") {
                    swap_free = val_kb * 1024ULL;
                }
            }
        }
        if (swap_total >= swap_free) {
            stats.swap_used_bytes = swap_total - swap_free;
        }
    }
#endif

    return stats;
}

} // namespace amuquant
