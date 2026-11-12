#ifndef KYTY_COMMON_DEBUG_COUNTERS_H_
#define KYTY_COMMON_DEBUG_COUNTERS_H_

#include <atomic>
#include <chrono>
#include <cstdint>

namespace Common::DebugCounters {

static_assert(std::atomic<uint64_t>::is_always_lock_free && std::atomic<bool>::is_always_lock_free);

// Collect while the panel or the performance log is enabled. Relaxed, lock-free
// atomics keep updates cheap and safe in signal handlers. Rates use two snapshots.
enum class Counter {
	// GPU work.
	Draws,
	DrawInstances,
	IndirectDraws,
	Dispatches,
	RenderPassStarts, // Native beginRendering commands; includes emulated mesh restarts.
	ComputePasses,    // Native guest compute dispatch commands; excludes driver tessellation.
	BufferBlits,      // Recorded buffer copies, not driver Metal encoder counts.
	BufferFills,      // MoltenVK can encode fills as compute instead of blits.
	RenderPassMerges,
	InPassClears,
	AttachmentStoresSkipped,
	QueueSubmits,
	PoolReclaimsDraw,
	PoolReclaimsFrame,
	PoolReclaimsPressure,
	PoolRecreateNs,
	TessellationDraws,
	TessellationVertices, // Known input vertices times instances, before tessellation expansion.
	TessellationIndirectDraws, // Vertex counts are GPU-only; excluded from TessellationVertices.
	TessellationPatches,
	TessellationRectangles,
	TessellationCorners,
	CornerFallbackIndirect,
	CornerFallbackRestart,
	CornerFallbackCaptureRange,
	CornerFallbackOutputKind,
	CornerFallbackSideEffects,
	CornerFallbackSwitchOff,
	CornerFallbackOther,
	VsStoreNativeDraws,
	MeshBufferDraws,
	GpuWaits, // CPU blocked on a GPU timeline semaphore.
	GpuWaitNs,
	GpuFaults, // Guest writes to GPU-tracked memory.
	ShadersCompiled,
	ShaderCompileNs,
	PipelinesCreated,
	PipelineCreateNs,
	// Caches.
	TextureUploadBytes,
	TextureDownloadBytes,
	TextureEvictions,
	ColorMetadataReadbacks, // CPU blocked reading back GPU-written DCC/CMASK metadata.
	ColorMetadataReadbackNs,
	ColorMetadataChecks, // GPU clear-key checks of DCC/CMASK metadata.
	BufferUploadBytes,
	BufferDownloadBytes,
	BufferEvictions,
	// Presentation.
	FlipsRequested,
	FlipsRejected, // Guest flip queue was full.
	LateVblanks,   // The present thread fell behind the vblank rate.
	SwapchainRecreations,
	DroppedPresents,
	// Emulator.
	EmulatedInstructions, // Host CPU lacked an instruction the guest used.
	UnresolvedImportCalls,
	GuestFileReadBytes,
	GuestFileWriteBytes,
	AudioUnderruns,
	// Overlapping host elapsed timings, including waits.
	GpuThreadBusyNs,
	PresentNs, // Includes image acquire and overlay preparation.
	PresentAcquireNs,
	OverlayNs,
	Count
};

enum class Peak { ShaderCompileNs, PipelineCreateNs, Count };

namespace Detail {
inline std::atomic<bool>     g_enabled {false};
inline std::atomic<bool>     g_perf_enabled {false};
inline std::atomic<uint64_t> g_counters[static_cast<int>(Counter::Count)] {};
inline std::atomic<uint64_t> g_peaks[static_cast<int>(Peak::Count)] {};
} // namespace Detail

[[nodiscard]] inline bool Enabled() noexcept {
	return Detail::g_enabled.load(std::memory_order_relaxed);
}

inline void SetEnabled(bool enabled) noexcept {
	Detail::g_enabled.store(enabled, std::memory_order_relaxed);
}

inline void SetPerfEnabled(bool enabled) noexcept {
	Detail::g_perf_enabled.store(enabled, std::memory_order_relaxed);
}

[[nodiscard]] inline bool Collecting() noexcept {
	return Enabled() || Detail::g_perf_enabled.load(std::memory_order_relaxed);
}

inline void Add(Counter counter, uint64_t value = 1) noexcept {
	if (!Collecting()) {
		return;
	}
	Detail::g_counters[static_cast<int>(counter)].fetch_add(value, std::memory_order_relaxed);
}

class ScopedTimer final {
public:
	explicit ScopedTimer(Counter duration): ScopedTimer(Counter::Count, duration) {}
	ScopedTimer(Counter count, Counter duration, Peak peak = Peak::Count)
	    : m_count(count), m_duration(duration), m_peak(peak),
	      m_start(Collecting() ? std::chrono::steady_clock::now()
	                           : std::chrono::steady_clock::time_point {}) {}
	~ScopedTimer() {
		if (m_start == std::chrono::steady_clock::time_point {} || !Collecting()) {
			return;
		}
		const auto elapsed =
		    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                              std::chrono::steady_clock::now() - m_start)
		                              .count());
		if (m_count != Counter::Count) {
			Add(m_count);
		}
		Add(m_duration, elapsed);
		if (m_peak != Peak::Count) {
			auto& peak     = Detail::g_peaks[static_cast<int>(m_peak)];
			auto  previous = peak.load(std::memory_order_relaxed);
			while (previous < elapsed &&
			       !peak.compare_exchange_weak(previous, elapsed, std::memory_order_relaxed)) {
			}
		}
	}
	ScopedTimer(const ScopedTimer&)            = delete;
	ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
	Counter                               m_count;
	Counter                               m_duration;
	Peak                                  m_peak;
	std::chrono::steady_clock::time_point m_start;
};

[[nodiscard]] inline uint64_t Get(Counter counter) noexcept {
	return Detail::g_counters[static_cast<int>(counter)].load(std::memory_order_relaxed);
}

[[nodiscard]] inline uint64_t Get(Peak peak) noexcept {
	return Detail::g_peaks[static_cast<int>(peak)].load(std::memory_order_relaxed);
}

} // namespace Common::DebugCounters

#endif /* KYTY_COMMON_DEBUG_COUNTERS_H_ */
