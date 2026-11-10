#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUWORKLOG_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUWORKLOG_H_

#include <cstdint>
#include <string>

// Diagnostics for GPU work that does not finish. Controlled by environment variables:
//   KYTY_GPU_TRACE=1     keep the last recorded draws and dispatches for the stall report.
//   KYTY_GPU_ISOLATE=1   implies TRACE; print each first-use or very large draw/dispatch and
//                        wait for the GPU to finish it before more work is recorded.
//   KYTY_GPU_SKIP=a,b    skip draws and dispatches that use a shader with one of these hashes.
namespace Libs::Graphics::GpuWorkLog {

struct Work {
	char     kind         = '?'; // 'D' draw, 'M' emulated mesh draw, 'C' dispatch, 'I' indirect dispatch
	uint64_t shaders[3]   = {};  // VS/MS, PS, CS
	uint64_t pipeline     = 0;   // host pipeline identity
	uint64_t instances    = 1;
	uint64_t elements     = 0;   // vertices or indices per instance; workgroups for a dispatch
	uint64_t host_passes  = 1;   // host render or compute passes this guest operation records
	uint32_t width        = 0;   // first color target or depth target
	uint32_t height       = 0;
	uint32_t color_count  = 0;
	bool     tessellation = false;
};

[[nodiscard]] bool TraceEnabled() noexcept;
[[nodiscard]] bool IsolateEnabled() noexcept;
// True when a shader hash is in KYTY_GPU_SKIP. Logs the first skip of each hash.
[[nodiscard]] bool ShouldSkip(uint64_t shader_hash) noexcept;

// Records work for the stall report. Returns true when isolation mode wants the caller to
// submit and wait for this operation by itself (first use of its pipeline, or very large).
bool Record(uint64_t tick, const Work& work);

// Called while a wait for the GPU takes too long.
void ReportStall(uint64_t waited_tick, uint64_t finished_tick, double seconds);

} // namespace Libs::Graphics::GpuWorkLog

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUWORKLOG_H_
