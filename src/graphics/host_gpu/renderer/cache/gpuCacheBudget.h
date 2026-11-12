#ifndef KYTY_GRAPHICS_GPU_CACHE_BUDGET_H_
#define KYTY_GRAPHICS_GPU_CACHE_BUDGET_H_

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

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

enum class CommandPoolRetirementAction { Pending, Release, Wait };

// A GPU timeline signal can precede MoltenVK's host completion handlers. A fence
// checkpoint is also required before those handlers' allocation pools can be freed.
struct CommandPoolRetirementBudget {
	static constexpr size_t   MaxPendingPools = 2;
	static constexpr uint64_t MaxPendingFrames = 2;

	[[nodiscard]] static constexpr CommandPoolRetirementAction Poll(
	    uint64_t last_tick, uint64_t completed_tick, bool fence_complete,
	    size_t pending_pools, uint64_t pending_frames) {
		if (last_tick <= completed_tick && fence_complete) {
			return CommandPoolRetirementAction::Release;
		}
		if (pending_pools >= MaxPendingPools || pending_frames >= MaxPendingFrames) {
			return CommandPoolRetirementAction::Wait;
		}
		return CommandPoolRetirementAction::Pending;
	}
};

struct TessellationReclaimPolicy {
	static constexpr uint64_t DefaultVertices = 512 * 1024;
	uint64_t vertices = DefaultVertices;
	uint64_t hard_vertices = DefaultVertices * 8;
	bool at_frame = false;

	// Zero and invalid values restore the default. Parse once before GPU work starts.
	[[nodiscard]] static TessellationReclaimPolicy Parse(const char* vertices_value,
	                                                    const char* hard_value,
	                                                    const char* frame_value) {
		const auto read = [](const char* text, uint64_t fallback) {
			if (text == nullptr) return fallback;
			const std::string_view value(text);
			uint64_t result = 0;
			const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
			return parsed.ec == std::errc {} && parsed.ptr == value.data() + value.size() && result != 0
			    ? result : fallback;
		};
		TessellationReclaimPolicy policy;
		policy.vertices = read(vertices_value, DefaultVertices);
		const auto hard_default = policy.vertices > std::numeric_limits<uint64_t>::max() / 8
		    ? std::numeric_limits<uint64_t>::max() : policy.vertices * 8;
		policy.hard_vertices = read(hard_value, hard_default);
		policy.at_frame = frame_value != nullptr && std::string_view(frame_value) == "1";
		return policy;
	}
};

// Bound the work retained by one MoltenVK pool without waiting at every small draw.
// This is a soft work limit, not a byte cap: one guest draw can exceed it. GPU-only
// indirect counts are retired at the frame boundary instead of forcing a reset per draw.
struct TessellationBatchBudget {
	static constexpr uint64_t MaxVertices = TessellationReclaimPolicy::DefaultVertices;
	// Three live ring pools retain less total work than one legacy pool, before
	// per-draw overshoot and the driver's allocation rounding.
	static constexpr uint64_t MaxRingVertices = 128 * 1024;

	explicit TessellationBatchBudget(TessellationReclaimPolicy policy = {}): m_policy(policy) {}

	void Record(uint32_t vertices, uint32_t instances) {
		// Keep the sum for the hard ceiling. Saturate before addition can overflow.
		const auto count = uint64_t {vertices} * instances;
		m_vertices += std::min(count, std::numeric_limits<uint64_t>::max() - m_vertices);
	}
	void RecordIndirect() { m_indirect = true; }
	[[nodiscard]] bool NeedsReclaim(bool use_ring = false) const {
		return m_vertices >= (use_ring ? MaxRingVertices : m_policy.vertices);
	}
	[[nodiscard]] bool NeedsDrawReclaim(bool use_ring = false) const {
		if (use_ring) return NeedsReclaim(true);
		return m_policy.at_frame ? m_vertices >= m_policy.hard_vertices : NeedsReclaim();
	}
	[[nodiscard]] bool NeedsFrameReclaim(bool bound_lifetime = false) const {
		// Ring policy does not use the legacy overrides.
		return m_indirect || (bound_lifetime ? m_vertices != 0 : NeedsReclaim());
	}
	void Reset() {
		m_vertices = 0;
		m_indirect = false;
	}

private:
	TessellationReclaimPolicy m_policy;
	uint64_t m_vertices = 0;
	bool     m_indirect = false;
};

[[nodiscard]] constexpr uint64_t TextureReclaimAge(CacheReclaimLevel level) {
	return level == CacheReclaimLevel::Critical ? 4
	       : level == CacheReclaimLevel::Pressure ? 8
	                                             : 16;
}

} // namespace Libs::Graphics

#endif // KYTY_GRAPHICS_GPU_CACHE_BUDGET_H_
