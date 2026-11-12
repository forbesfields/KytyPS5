#include "common/debugCounters.h"
#include "common/lruCache.h"
#include "graphics/host_gpu/renderer/cache/gpuCacheBudget.h"
#include "graphics/host_gpu/renderer/cache/streamBufferGrowth.h"

#include <cstdio>
#include <cstdlib>

namespace {

using Libs::Graphics::CacheReclaimLevel;
using Libs::Graphics::GpuCacheBudget;
using Libs::Graphics::TextureReclaimAge;
using Libs::Graphics::UnifiedGpuCacheLimit;

constexpr uint64_t MiB = 1024 * 1024;
constexpr uint64_t GiB = 1024 * MiB;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "GpuCacheBudgetTests: %s\n", message);
		std::exit(1);
	}
}

} // namespace

int main() {
  using Libs::Graphics::TryGrowUploadRing;
  Check(Libs::Graphics::UploadRingInitialSize(false) == 512 * MiB &&
            Libs::Graphics::UploadRingInitialSize(true) == 64 * MiB &&
            Libs::Graphics::UploadRingMaximumSize == 256 * MiB,
        "adaptive startup must save 448 MiB and the kill switch must restore 512 MiB");
  std::puts("Upload ring startup allocation: 512 -> 64 MiB; saving 448 MiB");
  Check(Libs::Graphics::UploadRingMaximumSize +
            Libs::Graphics::UploadRingMaximumSize / 2 +
            Libs::Graphics::UploadRingInitialSize(true) == 448 * MiB,
        "queued growth allocations must remain below the old fixed ring size");
  std::puts("Upload ring cap: 256 MiB; saving at least 256 MiB of fixed storage");
  uint64_t capacity = 64 * MiB;
  unsigned allocations = 0;
  const auto allocate = [&](uint64_t size) {
    ++allocations;
    capacity = size;
    return true;
  };
  Check(!TryGrowUploadRing(capacity, capacity, 256 * MiB, allocate) &&
            allocations == 0,
        "an exact fit must not grow the upload ring");
  Check(TryGrowUploadRing(capacity, capacity + 1, 256 * MiB, allocate) &&
            capacity == 128 * MiB && allocations == 1,
        "one byte beyond the ring must double its capacity");
  Check(TryGrowUploadRing(capacity, 200 * MiB, 256 * MiB, allocate) &&
            capacity == 256 * MiB && allocations == 2,
        "a large upload must grow directly to a fitting capacity");
  Check(!TryGrowUploadRing(capacity, capacity + 1, 256 * MiB, allocate) &&
            allocations == 2,
        "uploads above the cap must use the temporary-buffer fallback");
  capacity = 64 * MiB;
  const auto fail_allocate = [&](uint64_t size) {
    ++allocations;
    Check(size == 128 * MiB, "failed growth must request the next size");
    return false;
  };
  Check(!TryGrowUploadRing(capacity, capacity + 1, 256 * MiB, fail_allocate) &&
            capacity == 64 * MiB && allocations == 3,
        "failed growth must preserve capacity for ring reuse or "
        "temporary-buffer fallback");
  Check(!TryGrowUploadRing(capacity, capacity + 1, 0, allocate) &&
            allocations == 3,
        "the kill switch must disable all growth allocations");
  Check(TryGrowUploadRing(64 * MiB, 140 * MiB, 150 * MiB, allocate) &&
            capacity == 150 * MiB,
        "growth must stop at a non-power-of-two cap");
  Check(!TryGrowUploadRing(0, 1, UINT64_MAX, allocate),
        "an empty ring cannot double its size");
  Check(
      TryGrowUploadRing(UINT64_MAX / 2 + 1, UINT64_MAX, UINT64_MAX, allocate) &&
          capacity == UINT64_MAX,
      "growth near the address limit must not overflow");
  std::puts("Upload ring growth, boundaries, allocation failure, and "
            "fixed-size fallback: ok");
	const auto limit = UnifiedGpuCacheLimit(18 * GiB, 24 * GiB, 12736 * MiB);
	Check(limit == 7744 * MiB, "24 GiB host must reserve space for guest backing and macOS");
	Check(UnifiedGpuCacheLimit(6 * GiB, 24 * GiB, 12736 * MiB) == 6 * GiB,
	      "the driver can impose a lower limit");
	Check(UnifiedGpuCacheLimit(8 * GiB, 8 * GiB, 12 * GiB) == 0,
	      "a guest larger than host RAM must not wrap the limit");

	const GpuCacheBudget normal {0, limit, CacheReclaimLevel::Idle};
	Check(normal.Level(5 * GiB) == CacheReclaimLevel::Idle, "working set fits without eviction");
	Check(normal.Level(6 * GiB) == CacheReclaimLevel::Normal, "soft limit reclaims cold textures");
	Check(normal.Level(7 * GiB) == CacheReclaimLevel::Pressure, "near limit reclaims both caches");
	Check(normal.Level(limit) == CacheReclaimLevel::Critical, "limit triggers critical reclaim");
	const GpuCacheBudget warning {0, limit, CacheReclaimLevel::Pressure};
	Check(warning.Level(5 * GiB) == CacheReclaimLevel::Pressure,
	      "macOS warning triggers reclaim below the configured limit");
	Check(warning.Level(limit) == CacheReclaimLevel::Critical,
	      "host warning cannot weaken budget-triggered critical reclaim");
	const GpuCacheBudget critical {0, limit, CacheReclaimLevel::Critical};
	Check(critical.Level(5 * GiB) == CacheReclaimLevel::Critical,
	      "macOS critical pressure triggers critical reclaim below the limit");

	using namespace std::chrono_literals;
	// In the title-scene capture, the caches freed Vulkan blocks while Metal usage grew.
	const GpuCacheBudget scratch_pressure {10327 * MiB, limit, CacheReclaimLevel::Pressure};
	Check(scratch_pressure.ShouldReclaimDriverPool(3000 * MiB, 1s),
	      "a critical Metal spike can trigger another queue drain after one second");
	Check(!scratch_pressure.ShouldReclaimDriverPool(3000 * MiB, 999ms),
	      "critical pressure still limits the rate of queue drains");
	Check(!scratch_pressure.ShouldReclaimDriverPool(10000 * MiB, 10s),
	      "cache-owned memory pressure should not drain the GPU queue");
	const GpuCacheBudget warning_driver {7 * GiB, limit, CacheReclaimLevel::Pressure};
	Check(!warning_driver.ShouldReclaimDriverPool(3 * GiB, 4999ms) &&
	      warning_driver.ShouldReclaimDriverPool(3 * GiB, 5s),
	      "warning pressure permits one drain every five seconds");
	const GpuCacheBudget idle_driver {5 * GiB, limit, CacheReclaimLevel::Idle};
	Check(!idle_driver.ShouldReclaimDriverPool(3 * GiB, 10s), "do not drain an idle renderer");
	const GpuCacheBudget stale_usage {6 * GiB, limit, CacheReclaimLevel::Pressure};
	Check(!stale_usage.ShouldReclaimDriverPool(7 * GiB, 10s),
	      "a stale driver sample must not wrap the untracked byte count");

	using Libs::Graphics::CommandPoolRetirementAction;
	using Libs::Graphics::CommandPoolRetirementBudget;
	const auto poll = CommandPoolRetirementBudget::Poll;
	Check(poll(12, 12, true, 1, 0) == CommandPoolRetirementAction::Release,
	      "a completed tick and host fence permit pool destruction");
	Check(poll(12, 15, true, 1, 0) == CommandPoolRetirementAction::Release,
	      "later completed ticks also permit destruction");
	Check(poll(12, 11, false, 1, 0) == CommandPoolRetirementAction::Pending,
	      "a pending tick must keep its pool alive without blocking");
	Check(poll(12, 12, false, 1, 0) == CommandPoolRetirementAction::Pending,
	      "a GPU signal must not free a pool before its host completion fence");
	Check(poll(12, 11, true, 1, 0) == CommandPoolRetirementAction::Pending,
	      "a stale tick sample must be refreshed before pool destruction");
	Check(poll(12, 11, false, CommandPoolRetirementBudget::MaxPendingPools, 0) ==
	          CommandPoolRetirementAction::Wait,
	      "ring exhaustion waits for the oldest pool instead of growing memory");
	Check(poll(12, 12, false, CommandPoolRetirementBudget::MaxPendingPools, 0) ==
	          CommandPoolRetirementAction::Wait,
	      "ring exhaustion must wait for host completion even after the GPU signal");
	Check(poll(12, 12, true, CommandPoolRetirementBudget::MaxPendingPools, 0) ==
	          CommandPoolRetirementAction::Release,
	      "a full ring with a completed oldest pool needs no wait");
	Check(poll(12, 11, false, 1, CommandPoolRetirementBudget::MaxPendingFrames - 1) ==
	          CommandPoolRetirementAction::Pending,
	      "a recent retired pool can overlap GPU execution");
	Check(poll(12, 11, false, 1, CommandPoolRetirementBudget::MaxPendingFrames) ==
	          CommandPoolRetirementAction::Wait,
	      "an old pool must drain even if no new pool is retired");
	Check(poll(12, 12, false, 1, CommandPoolRetirementBudget::MaxPendingFrames) ==
	          CommandPoolRetirementAction::Wait,
	      "the frame bound also covers delayed host completion handlers");
	Check(poll(12, 12, true, 1, CommandPoolRetirementBudget::MaxPendingFrames) ==
	          CommandPoolRetirementAction::Release,
	      "an old completed pool is released without a wait");

	Libs::Graphics::TessellationBatchBudget batch;
	Check(!batch.NeedsReclaim(), "an empty pool needs no tessellation reclaim");
	Check(!batch.NeedsFrameReclaim(true), "an empty ring pool needs no frame retirement");
	batch.Record(3, 1);
	Check(!batch.NeedsFrameReclaim() && batch.NeedsFrameReclaim(true),
	      "the ring bounds small draws to one active frame; legacy keeps the old threshold");
	batch.Reset();
	Check(!batch.NeedsFrameReclaim(true), "retirement clears small draws from the frame budget");
	batch.Record(Libs::Graphics::TessellationBatchBudget::MaxRingVertices - 1, 1);
	Check(!batch.NeedsReclaim(true), "ring work below the limit stays in the active pool");
	batch.Record(1, 1);
	Check(batch.NeedsReclaim(true) && !batch.NeedsReclaim(),
	      "ring pools retire at 128K vertices while legacy pools keep the 512K limit");
	Check(Libs::Graphics::TessellationBatchBudget::MaxRingVertices *
	          (CommandPoolRetirementBudget::MaxPendingPools + 1) <=
	          Libs::Graphics::TessellationBatchBudget::MaxVertices,
	      "the sum of ring work targets must fit within the old work target");
	batch.Reset();
	for (unsigned draw = 0; draw < 18; ++draw) {
		batch.Record(28416, 1);
	}
	Check(!batch.NeedsReclaim(), "eighteen captured draws fit in the tessellation batch");
	batch.Record(28416, 1);
	Check(batch.NeedsReclaim(), "a nineteenth draw schedules retirement at the next boundary");
	batch.Reset();
	Check(!batch.NeedsFrameReclaim(), "pool retirement resets accumulated tessellation work");
	for (unsigned draw = 0; draw < 100; ++draw) batch.RecordIndirect();
	Check(!batch.NeedsReclaim() && batch.NeedsFrameReclaim(),
	      "GPU-only indirect counts retire once per frame, not once per draw");
	batch.Reset();
	Check(!batch.NeedsFrameReclaim(), "pool retirement clears indirect work too");
	batch.Record(28416, 19);
	Check(batch.NeedsReclaim(), "all instances count towards the pool lifetime");
	batch.Reset();
	batch.Record(UINT32_MAX, UINT32_MAX);
	batch.Record(UINT32_MAX, UINT32_MAX);
	Check(batch.NeedsReclaim(), "very large repeated draws must not overflow accounting");
	batch.Reset();
	batch.Record(UINT32_MAX, 0);
	Check(!batch.NeedsReclaim(), "zero instances allocate no tessellation output");

	using Libs::Graphics::TessellationBatchBudget;
	using Libs::Graphics::TessellationReclaimPolicy;
	const auto defaults = TessellationReclaimPolicy::Parse(nullptr, nullptr, nullptr);
	Check(defaults.vertices == 524288 && defaults.hard_vertices == 4194304 && !defaults.at_frame,
	      "unset switches keep 512K and immediate draw reclaim");
	const auto zero = TessellationReclaimPolicy::Parse("0", "0", "0");
	Check(zero.vertices == 524288 && zero.hard_vertices == 4194304 && !zero.at_frame,
	      "zero restores the default limits and disables frame deferral");
	for (const auto* invalid: {"", "-1", "+1", "1x", " 2", "18446744073709551616"}) {
		const auto policy = TessellationReclaimPolicy::Parse(invalid, invalid, invalid);
		Check(policy.vertices == 524288 && policy.hard_vertices == 4194304 && !policy.at_frame,
		      "invalid limits must not disable the work bound");
	}
	const auto override = TessellationReclaimPolicy::Parse("2097152", nullptr, "0");
	Check(override.vertices == 2097152 && override.hard_vertices == 16777216,
	      "the hard default must follow the threshold override");
	TessellationBatchBudget larger(override);
	larger.Record(2097151, 1);
	Check(!larger.NeedsDrawReclaim() && !larger.NeedsFrameReclaim(), "2M override below threshold");
	larger.Record(1, 1);
	Check(larger.NeedsDrawReclaim() && larger.NeedsFrameReclaim(), "2M override exact threshold");
	const auto deferred = TessellationReclaimPolicy::Parse("524288", nullptr, "1");
	TessellationBatchBudget frame_batch(deferred);
	frame_batch.Record(524288, 1);
	Check(frame_batch.NeedsReclaim() && !frame_batch.NeedsDrawReclaim() && frame_batch.NeedsFrameReclaim(),
	      "threshold work must wait for the next frame boundary");
	frame_batch.Record(3670015, 1);
	Check(!frame_batch.NeedsDrawReclaim(), "frame deferral below the hard ceiling");
	frame_batch.Record(1, 1);
	Check(frame_batch.NeedsDrawReclaim(), "the 8x hard ceiling must drain before another draw");
	frame_batch.Reset();
	Check(!frame_batch.NeedsDrawReclaim() && !frame_batch.NeedsFrameReclaim(),
	      "a frame or pressure reclaim must clear deferred work");
	frame_batch.RecordIndirect();
	Check(frame_batch.NeedsFrameReclaim() && !frame_batch.NeedsDrawReclaim(),
	      "indirect work must still reclaim at the frame boundary");
	TessellationBatchBudget hard_override(TessellationReclaimPolicy::Parse("100", "300", "1"));
	hard_override.Record(299, 1);
	Check(!hard_override.NeedsDrawReclaim(), "custom hard ceiling below boundary");
	hard_override.Record(1, 1);
	Check(hard_override.NeedsDrawReclaim(), "custom hard ceiling exact boundary");
	const auto huge_policy = TessellationReclaimPolicy::Parse("18446744073709551615", nullptr, "1");
	Check(huge_policy.hard_vertices == UINT64_MAX, "8x hard default must saturate");
	TessellationBatchBudget huge(huge_policy);
	huge.Record(UINT32_MAX, UINT32_MAX);
	Check(!huge.NeedsDrawReclaim(), "one maximal draw stays below UINT64_MAX");
	huge.Record(UINT32_MAX, UINT32_MAX);
	Check(huge.NeedsDrawReclaim(), "repeated maximal draws saturate without wrap");

	// Compare the old saturated policy with the production default over mixed work.
	for (const bool ring: {false, true}) {
		TessellationBatchBudget current(defaults);
		TessellationBatchBudget tuned_ring(TessellationReclaimPolicy::Parse("1", "2", "1"));
		uint64_t old_vertices = 0;
		bool old_indirect = false;
		for (unsigned draw = 1; draw <= 10000; ++draw) {
			const bool old_draw_reclaim = old_vertices >= (ring ? 131072u : 524288u);
			Check(current.NeedsDrawReclaim(ring) == old_draw_reclaim,
			      "unset draw reclaim must equal the base policy");
			if (ring) Check(tuned_ring.NeedsDrawReclaim(true) == old_draw_reclaim,
			                "legacy overrides must not change ring draw policy");
			if (old_draw_reclaim) {
				old_vertices = 0;
				old_indirect = false;
				current.Reset();
				tuned_ring.Reset();
			}
			if (draw % 13 == 0) {
				old_indirect = true;
				current.RecordIndirect();
				tuned_ring.RecordIndirect();
			} else {
				const uint32_t vertices = draw % 7 == 0 ? UINT32_MAX : draw * 37;
				const uint32_t instances = draw % 5;
				old_vertices = std::min<uint64_t>(524288,
				    old_vertices + std::min<uint64_t>(524288, uint64_t {vertices} * instances));
				current.Record(vertices, instances);
				tuned_ring.Record(vertices, instances);
			}
			if (draw % 19 == 0) {
				const bool old_frame_reclaim = old_indirect || old_vertices >= 524288 ||
				    (ring && old_vertices != 0);
				Check(current.NeedsFrameReclaim(ring) == old_frame_reclaim,
				      "unset frame reclaim must equal the base policy");
				if (ring) Check(tuned_ring.NeedsFrameReclaim(true) == old_frame_reclaim,
				                "legacy overrides must not change ring frame policy");
				if (old_frame_reclaim) {
					old_vertices = 0;
					old_indirect = false;
					current.Reset();
					tuned_ring.Reset();
				}
			}
		}
	}
	TessellationBatchBudget hard_disabled(TessellationReclaimPolicy::Parse("2097152", "1", "0"));
	hard_disabled.Record(524288, 1);
	Check(!hard_disabled.NeedsDrawReclaim(), "the hard override has no effect with deferral off");
	TessellationBatchBudget eight_million(TessellationReclaimPolicy::Parse("8388608", nullptr, nullptr));
	eight_million.Record(8388607, 1);
	Check(!eight_million.NeedsDrawReclaim(), "8M override below threshold");
	eight_million.Record(1, 1);
	Check(eight_million.NeedsDrawReclaim(), "8M override exact threshold");
	const auto count_reclaims = [](TessellationReclaimPolicy policy) {
		TessellationBatchBudget work(policy);
		unsigned reclaims = 0;
		for (unsigned frame = 0; frame < 8; ++frame) {
			for (unsigned draw = 0; draw < 80; ++draw) {
				if (work.NeedsDrawReclaim()) { ++reclaims; work.Reset(); }
				work.Record(28416, 1);
			}
			if (work.NeedsFrameReclaim()) { ++reclaims; work.Reset(); }
		}
		return reclaims;
	};
	Check(count_reclaims(defaults) == 33 && count_reclaims(deferred) == 8,
	      "640 small draws across eight frames must defer 33 drains into eight");
	std::puts("Reclaim fixture: 640 draws / 8 frames, default=33 drains, frame deferral=8 drains");
	std::puts("Reclaim policy: defaults match base for 20000 mixed draws; overrides, frame, hard ceiling: ok");
	using namespace Common::DebugCounters;
	SetEnabled(false);
	SetPerfEnabled(false);
	const auto before_counter = Get(Counter::PoolReclaimsDraw);
	Add(Counter::PoolReclaimsDraw);
	Check(Get(Counter::PoolReclaimsDraw) == before_counter, "disabled collectors must not count");
	SetPerfEnabled(true);
	Add(Counter::PoolReclaimsDraw, 3);
	Check(!Enabled() && Get(Counter::PoolReclaimsDraw) == before_counter + 3,
	      "performance log must count while the overlay is hidden");
	SetEnabled(true);
	SetEnabled(false);
	Add(Counter::PoolReclaimsDraw);
	Check(Get(Counter::PoolReclaimsDraw) == before_counter + 4,
	      "hiding the overlay must not stop performance log counters");
	SetPerfEnabled(false);
	std::puts("Performance counters: log collection is independent of overlay visibility: ok");

	Common::LeastRecentlyUsedCache<int, uint64_t> textures;
	(void)textures.Insert(1, 12);
	(void)textures.Insert(2, 30);
	(void)textures.Insert(3, 35);
	(void)textures.Insert(4, 39);
	const auto count_eligible = [&](CacheReclaimLevel level) {
		int count = 0;
		textures.ForEachItemBelow(40 - TextureReclaimAge(level), [&](int) { ++count; });
		return count;
	};
	Check(count_eligible(CacheReclaimLevel::Normal) == 1, "normal reclaim keeps warm textures");
	Check(count_eligible(CacheReclaimLevel::Pressure) == 2, "pressure adds eligible textures");
	Check(count_eligible(CacheReclaimLevel::Critical) == 3, "critical keeps recently used textures");
	return 0;
}
