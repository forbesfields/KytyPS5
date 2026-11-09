#pragma once

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <cstdint>
#include <vector>

namespace Libs::Graphics {

// Above all four groups of native shader resource bindings. Both compute and vertex stages
// access this buffer; it is private to the renderer, never a guest descriptor.
inline constexpr uint32_t MeshEmulationBinding = 255u;
static_assert(MeshEmulationBinding >= 4u *
    static_cast<uint32_t>(ShaderRecompiler::IR::DescriptorBindingKind::Count));

struct MeshEmulationField {
    ShaderRecompiler::IR::StageOutput output;
    uint32_t offset = 0;
};

struct MeshEmulationLayout {
    // Global header: guest group X base, instance base, total groups X, total instances.
    static constexpr uint32_t HeaderDwords = 4;
    uint32_t max_vertices = 0;
    uint32_t max_primitives = 0;
    uint32_t vertex_dwords = 0;
    uint32_t provoking_vertex = 0;
    std::vector<MeshEmulationField> fields;

    [[nodiscard]] uint32_t VerticesOffset() const { return 2u + max_primitives; }
    [[nodiscard]] uint32_t GroupDwords() const {
        return VerticesOffset() + max_vertices * vertex_dwords;
    }
};

[[nodiscard]] MeshEmulationLayout BuildMeshEmulationLayout(
    const std::vector<ShaderRecompiler::IR::StageOutput>& outputs,
    const ShaderVertexInputInfo& input);

[[nodiscard]] std::vector<uint32_t> BuildMeshEmulationVertexShader(
    const MeshEmulationLayout& layout, uint32_t provoking_vertex);

} // namespace Libs::Graphics
