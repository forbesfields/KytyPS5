#include "graphics/host_gpu/renderer/gpuWorkLog.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics::GpuWorkLog {

namespace {

constexpr size_t   RING_SIZE = 512;
// Work above these sizes is printed and isolated even after its first use.
constexpr uint64_t LARGE_HOST_PASSES = 256;
constexpr uint64_t LARGE_ELEMENTS    = uint64_t {1} << 24;

struct Entry {
	uint64_t tick = 0;
	Work     work;
};

bool EnvFlag(const char* name) {
	const char* value = std::getenv(name);
	return value != nullptr && value[0] != '\0' && value[0] != '0';
}

struct State {
	bool                         trace   = false;
	bool                         isolate = false;
	std::unordered_set<uint64_t> skip;
	std::mutex                   mutex;
	std::array<Entry, RING_SIZE> ring {};
	size_t                       next = 0;
	std::unordered_set<uint64_t> seen_pipelines;
	std::unordered_set<uint64_t> skipped_logged;

	State() {
		isolate = EnvFlag("KYTY_GPU_ISOLATE");
		trace   = isolate || EnvFlag("KYTY_GPU_TRACE");
		if (const char* list = std::getenv("KYTY_GPU_SKIP"); list != nullptr) {
			const char* cursor = list;
			while (*cursor != '\0') {
				char*      end  = nullptr;
				const auto hash = std::strtoull(cursor, &end, 16);
				if (end == cursor) {
					break;
				}
				skip.insert(hash);
				cursor = (*end == ',') ? end + 1 : end;
			}
		}
		if (trace || !skip.empty()) {
			std::printf("GPU work log: trace=%d isolate=%d skip=%zu shader(s)\n", trace, isolate,
			            skip.size());
			std::fflush(stdout);
		}
	}
};

State& Get() {
	static State state;
	return state;
}

void Print(const char* prefix, uint64_t tick, const Work& w) {
	std::printf("%s tick=%" PRIu64 " kind=%c vs=0x%016" PRIx64 " ps=0x%016" PRIx64
	            " cs=0x%016" PRIx64 " instances=%" PRIu64 " elements=%" PRIu64
	            " host_passes=%" PRIu64 " target=%ux%u colors=%u tess=%d\n",
	            prefix, tick, w.kind, w.shaders[0], w.shaders[1], w.shaders[2], w.instances,
	            w.elements, w.host_passes, w.width, w.height, w.color_count, w.tessellation);
	std::fflush(stdout);
}

} // namespace

bool TraceEnabled() noexcept {
	return Get().trace;
}

bool IsolateEnabled() noexcept {
	return Get().isolate;
}

bool ShouldSkip(uint64_t shader_hash) noexcept {
	auto& s = Get();
	if (s.skip.empty() || shader_hash == 0 || !s.skip.contains(shader_hash)) {
		return false;
	}
	std::lock_guard lock(s.mutex);
	if (s.skipped_logged.insert(shader_hash).second) {
		std::printf("GPU work log: skipping work that uses shader 0x%016" PRIx64 "\n", shader_hash);
		std::fflush(stdout);
	}
	return true;
}

bool Record(uint64_t tick, const Work& work) {
	auto& s = Get();
	if (!s.trace) {
		return false;
	}
	bool first_use = false;
	{
		std::lock_guard lock(s.mutex);
		s.ring[s.next] = {tick, work};
		s.next         = (s.next + 1) % RING_SIZE;
		first_use      = work.pipeline != 0 && s.seen_pipelines.insert(work.pipeline).second;
	}
	const bool large = work.host_passes >= LARGE_HOST_PASSES ||
	                   work.instances * std::max<uint64_t>(work.elements, 1) >= LARGE_ELEMENTS;
	if (!s.isolate || (!first_use && !large)) {
		return false;
	}
	Print(first_use ? "GPU isolate (first use):" : "GPU isolate (large):", tick, work);
	return true;
}

void ReportStall(uint64_t waited_tick, uint64_t finished_tick, double seconds) {
	auto& s = Get();
	std::printf("GPU stall: no completion for %.0f s; waiting for submission %" PRIu64
	            ", last finished %" PRIu64 "%s\n",
	            seconds, waited_tick, finished_tick,
	            s.trace ? "" : " (set KYTY_GPU_TRACE=1 to list the unfinished work)");
	if (s.trace) {
		std::vector<Entry> pending;
		{
			std::lock_guard lock(s.mutex);
			for (size_t i = 0; i < RING_SIZE; ++i) {
				const auto& entry = s.ring[(s.next + i) % RING_SIZE];
				if (entry.tick > finished_tick && entry.tick <= waited_tick) {
					pending.push_back(entry);
				}
			}
		}
		std::printf("GPU stall: %zu recorded operation(s) in unfinished submissions:\n",
		            pending.size());
		for (const auto& entry: pending) {
			Print("  unfinished:", entry.tick, entry.work);
		}
	}
	std::fflush(stdout);
}

} // namespace Libs::Graphics::GpuWorkLog
