#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_STREAMBUFFERGROWTH_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_STREAMBUFFERGROWTH_H_

#include <algorithm>
#include <cstdint>

namespace Libs::Graphics {

// Even with both retired rings alive, 64 + 128 + 256 MiB stays below the old 512 MiB ring.
inline constexpr uint64_t UploadRingMaximumSize = 256ull * 1024 * 1024;

[[nodiscard]] constexpr uint64_t UploadRingInitialSize(bool adaptive) {
	return (adaptive ? 64ull : 512ull) * 1024 * 1024;
}

// A failed allocation leaves the caller's current ring intact.
template <typename Allocate>
[[nodiscard]] bool TryGrowUploadRing(uint64_t current, uint64_t required, uint64_t maximum,
                                     Allocate&& allocate) {
	if (current == 0 || required <= current || required > maximum || current >= maximum) {
		return false;
	}
	auto next = current;
	while (next < required) {
		next += std::min(next, maximum - next);
	}
	return allocate(next);
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_STREAMBUFFERGROWTH_H_
