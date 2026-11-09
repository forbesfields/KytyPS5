#ifndef EMULATOR_SRC_GRAPHICS_SHADER_TRIANGLESHADER_H_
#define EMULATOR_SRC_GRAPHICS_SHADER_TRIANGLESHADER_H_

#include "graphics/shader/rectListShader.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shader.h"

namespace Libs::Graphics {

struct TriangleParameter {
    uint32_t input_location = 0;
    uint32_t output_location = 0;
    bool per_vertex = false;
    bool flat = false;
    bool no_perspective = false;
    ShaderRecompiler::IR::StageInputKind source = ShaderRecompiler::IR::StageInputKind::Parameter;
};

// Flat arrays carry the original register bits from all three triangle corners.
std::vector<TriangleParameter> TriangleParameters(
    const ShaderPixelInputInfo& pixel, const ShaderRecompiler::IR::ShaderInfo& info);
std::vector<uint32_t> ExpandTriangleIndices(Prospero::PrimitiveType primitive,
                                          std::span<const uint32_t> indices,
                                          bool restart, uint32_t restart_index,
                                          bool provoking_last = false);
RectListShaders BuildTriangleShaders(std::span<const TriangleParameter> parameters,
                                    uint32_t clip_distances, uint32_t provoking_vertex);

} // namespace Libs::Graphics

#endif
