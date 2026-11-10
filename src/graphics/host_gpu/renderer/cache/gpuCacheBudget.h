#ifndef KYTY_GRAPHICS_GPU_CACHE_BUDGET_H_
#define KYTY_GRAPHICS_GPU_CACHE_BUDGET_H_

#include <algorithm>
#include <chrono>
#include <cstdint>

namespace Libs::Graphics {

enum class CacheReclaimLevel { Idle, Normal, Pressure, Critical };

[[nodiscard]] constexpr uint64_t SubtractBytes(uint64_t bytes, uint64_t used) {
	return bytes - std::min(bytes, used);
}

// The guest can use all its backing memory. Keep room for it and for other host processes.
[[nodiscard]] constexpr uint64_t UnifiedGpuCacheLimit(uint64_t driver_limit, uint64_t host_ram,
                                                      uint64_t guest_capacity) {
	constexpr uint64_t host_reserve = 4ull * 1024 * 1024 * 1024;
	return std::min(driver_limit, SubtractBytes(SubtractBytes(host_ram, guest_capacity), host_reserve));
}

// Both caches see the same whole-process GPU usage and the same host pressure per frame.
struct GpuCacheBudget {
	uint64_t          used;
	uint64_t          limit;
	CacheReclaimLevel host_level;

	[[nodiscard]] constexpr CacheReclaimLevel Level(uint64_t usage) const {
		const auto level = usage >= limit              ? CacheReclaimLevel::Critical
		                   : usage >= limit - limit / 8 ? CacheReclaimLevel::Pressure
		                   : usage >= limit - limit / 4 ? CacheReclaimLevel::Normal
		                                               : CacheReclaimLevel::Idle;
		return std::max(level, host_level);
	}

	[[nodiscard]] constexpr bool ShouldReclaimDriverPool(uint64_t allocator_blocks,
	                                                     std::chrono::steady_clock::duration elapsed) const {
		using namespace std::chrono_literals;
		constexpr uint64_t min_driver_overhead = 1024ull * 1024 * 1024;
		const auto level = Level(used);
		const auto interval = level == CacheReclaimLevel::Critical ? 1s : 5s;
		return level >= CacheReclaimLevel::Pressure && elapsed >= interval &&
		       SubtractBytes(used, allocator_blocks) >= min_driver_overhead;
	}
};

[[nodiscard]] constexpr uint64_t TextureReclaimAge(CacheReclaimLevel level) {
	return level == CacheReclaimLevel::Critical ? 4
	       : level == CacheReclaimLevel::Pressure ? 8
	                                             : 16;
}

} // namespace Libs::Graphics

#endif // KYTY_GRAPHICS_GPU_CACHE_BUDGET_H_
