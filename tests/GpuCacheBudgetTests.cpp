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

	Libs::Graphics::TessellationBatchBudget batch;
	Check(!batch.NeedsReclaim(), "an empty pool needs no tessellation reclaim");
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
