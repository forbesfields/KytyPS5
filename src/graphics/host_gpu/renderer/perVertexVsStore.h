#pragma once

#include "graphics/shader/meshEmulation.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace Libs::Graphics {

// Shader cache modes: 0 uses tessellation, 1 uses mesh records, 2 uses native capture.
uint32_t PerVertexBufferMode(bool emulated_mesh, bool native_candidate);

// A separate vertex-only draw produces these records. A command buffer barrier makes them
// visible to the consumer VS and PS. This is portable Vulkan; same-draw vertex stores followed
// by fragment loads would depend on driver scheduling, even on an Apple tile renderer.
inline constexpr uint32_t VertexStoreSet = 2u;
inline constexpr uint32_t VertexStoreHeaderDwords = 8u;
enum VertexStoreHeader : uint32_t {
    VertexStoreFirst = 0, VertexStoreSpan = 1, VertexStoreFirstInstance = 2,
    VertexStoreIndexCount = 3, VertexStoreStride = 4, VertexStoreRecords = 5,
};

struct VertexStorePlan {
    bool valid = false;
    uint32_t first = 0;
    uint32_t span = 0;
    uint32_t records_offset = 0;
    uint64_t bytes = 0;
    std::vector<uint32_t> indices; // relative vertex indices, one DWORD per corner
};

// Like the upstream VS-store range scan, include the signed base vertex before planning storage.
// The 64-bit checks keep record arithmetic and the storage descriptor inside their bounds.
inline VertexStorePlan PlanVertexStore(std::span<const uint32_t> indices, int32_t vertex_offset,
                                      uint32_t instances, uint32_t stride, uint64_t byte_limit) {
    VertexStorePlan plan;
    if (indices.empty() || indices.size() % 3u != 0 || indices.size() > UINT32_MAX - VertexStoreHeaderDwords ||
        instances == 0 || stride == 0) return plan;
    uint32_t first = UINT32_MAX, last = 0;
    for (const auto index : indices) {
        const int64_t vertex = int64_t{index} + vertex_offset;
        if (vertex < 0 || vertex > INT32_MAX) return plan;
        first = std::min(first, static_cast<uint32_t>(vertex));
        last = std::max(last, static_cast<uint32_t>(vertex));
    }
    const uint64_t span = uint64_t{last} - first + 1u;
    const uint64_t records = span * instances;
    const uint64_t header = VertexStoreHeaderDwords + indices.size();
    if (records > (UINT32_MAX - header) / stride) return plan;
    const auto bytes = (header + records * stride) * sizeof(uint32_t);
    if (bytes > byte_limit) return plan;
    plan.valid = true;
    plan.first = first;
    plan.span = static_cast<uint32_t>(span);
    plan.records_offset = static_cast<uint32_t>(header);
    plan.bytes = bytes;
    plan.indices.reserve(indices.size());
    for (const auto index : indices)
        plan.indices.push_back(static_cast<uint32_t>(int64_t{index} + vertex_offset) - first);
    return plan;
}

// The same record layout is used by the capture emitter and the generated consumer VS.
inline MeshEmulationLayout BuildVertexStoreLayout(
    const std::vector<ShaderRecompiler::IR::StageOutput>& outputs, uint32_t user_clip_planes = 0) {
    using namespace ShaderRecompiler::IR;
    MeshEmulationLayout layout;
    const auto add = [&](const StageOutput& output) {
        layout.fields.push_back({output, layout.vertex_dwords});
        layout.vertex_dwords += output.kind == StageOutputKind::Position ||
                                output.kind == StageOutputKind::Parameter ? 4u : 1u;
    };
    for (const auto& output : outputs) add(output);
    const auto original = CountOutputDistances(ShaderType::Mesh, outputs);
    const auto distances = CountOutputDistances(ShaderType::Vertex, outputs, user_clip_planes);
    for (uint32_t index = original.clip; index < distances.clip; ++index) {
        if (std::ranges::none_of(outputs, [&](const auto& output) {
                return output.kind == StageOutputKind::ClipDistance && output.index == index;
            })) add({StageOutputKind::ClipDistance, index, 0u, "clip"});
    }
    return layout;
}

} // namespace Libs::Graphics
