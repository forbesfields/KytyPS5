#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

// Host device capabilities and user options that change the emitted SPIR-V. Built once per device.
struct Profile {
	bool maintenance8         = false; // Dynamic texel offsets on image samples.
	bool native_float_min_max = false; // NMin/NMax give the guest's NaN and signed-zero results.
	// OpReadClockKHR Device tick in picoseconds; 0 without shaderDeviceClock.
	uint32_t device_clock_period_ps = 0;
	// Storage buffer elements past the descriptor range, to the byte, read zero and drop writes.
	bool robust_buffer_access = false;
	// 8 and 8_8 sRGB textures sample through sRGB views; without one the shader decodes them.
	bool r8_srgb_views   = false;
	bool r8g8_srgb_views = false;
	// Native float ops: NaN, signed-zero and denormal results may differ from the guest.
	bool relaxed_float = false;
	// Host stages with subgroup operations. A stage without them runs each invocation as a
	// one-lane wave: lane 0 is the only lane, so ballots, votes and lane reads need no host
	// subgroup. MoltenVK has them in fragment, compute and tessellation control stages only.
	bool vertex_subgroups                  = true;
	bool tessellation_evaluation_subgroups = true;
};

inline constexpr Profile DefaultProfile {};

// The stage runs as one-lane waves: the host stage has no subgroup operations.
[[nodiscard]] constexpr bool SingleLaneStage(const Profile& profile, ShaderType stage) {
	switch (stage) {
		case ShaderType::Vertex:
		case ShaderType::Local: return !profile.vertex_subgroups;
		case ShaderType::TessellationEvaluation: return !profile.tessellation_evaluation_subgroups;
		default: return false;
	}
}

// Raw corner inputs are flat DWORD addresses. A separate producer and command barrier must
// finish before fragment reads. The native capture path uses private descriptor set 2.
struct PvStoreInfo {
	bool per_vertex_buffer = false;
	bool vertex_store = false;
};
inline constexpr PvStoreInfo DefaultPvStoreInfo{};

// Resets and writes backend IDs and descriptor indices; do not emit the same program concurrently.
std::vector<uint32_t> EmitProgram(IR::Program& program, ShaderStageInputInfo input_info,
                                  uint32_t       push_data_start_dword = 0,
                                  const Profile& profile               = DefaultProfile,
                                  bool           emit_debug_info       = false,
                                  const PvStoreInfo& pv_store           = DefaultPvStoreInfo);

} // namespace Libs::Graphics::ShaderRecompiler::Spirv

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_ */
