#include "common/lruCache.h"
#include "graphics/host_gpu/renderer/cache/gpuCacheBudget.h"

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
