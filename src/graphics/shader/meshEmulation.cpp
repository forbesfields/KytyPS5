#include "graphics/shader/meshEmulation.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvBuilder.h"

#include <algorithm>
#include <array>
#include <utility>

namespace Libs::Graphics {

MeshEmulationLayout BuildMeshEmulationLayout(
    const std::vector<ShaderRecompiler::IR::StageOutput>& outputs,
    const ShaderVertexInputInfo& input) {
    using namespace ShaderRecompiler::IR;
    MeshEmulationLayout layout{.max_vertices = input.mesh.max_vertices,
                               .max_primitives = input.mesh.max_primitives,
                               .provoking_vertex = input.mesh.provoking_vertex};
    const auto add = [&](const StageOutput& output) {
        layout.fields.push_back({output, layout.vertex_dwords});
        switch (output.kind) {
            case StageOutputKind::Position:
            case StageOutputKind::Parameter: layout.vertex_dwords += 4u; break;
            case StageOutputKind::Layer:
            case StageOutputKind::ClipDistance:
            case StageOutputKind::CullDistance: layout.vertex_dwords++; break;
            default: EXIT("unsupported emulated mesh output kind=%u\n",
                          static_cast<uint32_t>(output.kind));
        }
    };
    for (const auto& output : outputs) add(output);
    const auto distances = CountOutputDistances(ShaderType::Mesh, outputs, input.user_clip_planes);
    for (uint32_t i = 0; i < distances.user_clip_planes; ++i)
        add({StageOutputKind::ClipDistance, i, 0u, "gl_ClipDistance"});
    EXIT_IF(layout.max_vertices == 0u || layout.max_primitives == 0u || layout.vertex_dwords == 0u);
    return layout;
}

namespace {

class MeshVertexEmitter {
public:
    MeshVertexEmitter(const MeshEmulationLayout& layout_, uint32_t provoking_)
        : layout(layout_), provoking(provoking_) {
        builder.RequireCapability(spv::CapabilityShader);
        builder.AddMemoryModel(spv::AddressingModelLogical, spv::MemoryModelGLSL450);
        void_type = builder.Type(spv::OpTypeVoid);
        bool_type = builder.Type(spv::OpTypeBool);
        uint_type = builder.Type(spv::OpTypeInt, 32u, 0u);
        int_type = builder.Type(spv::OpTypeInt, 32u, 1u);
        float_type = builder.Type(spv::OpTypeFloat, 32u);
        vec4_type = builder.Type(spv::OpTypeVector, float_type, 4u);
        const auto array = builder.DecoratedType(spv::OpTypeRuntimeArray,
            {{spv::OpDecorate, {spv::DecorationArrayStride, 4u}}}, uint_type);
        const auto block = builder.DecoratedType(spv::OpTypeStruct,
            {{spv::OpMemberDecorate, {0u, spv::DecorationOffset, 0u}},
             {spv::OpDecorate, {spv::DecorationBlock}}}, array);
        storage = builder.DefineGlobalVariable(Pointer(spv::StorageClassStorageBuffer, block),
                                               spv::StorageClassStorageBuffer);
        builder.AddAnnotation(spv::OpDecorate, storage, spv::DecorationDescriptorSet, 0u);
        builder.AddAnnotation(spv::OpDecorate, storage, spv::DecorationBinding, MeshEmulationBinding);
        builder.AddAnnotation(spv::OpDecorate, storage, spv::DecorationNonWritable);
        interface.push_back(storage);
        vertex_index = builder.DefineGlobalVariable(Pointer(spv::StorageClassInput, int_type),
                                                    spv::StorageClassInput);
        builder.AddAnnotation(spv::OpDecorate, vertex_index, spv::DecorationBuiltIn,
                              spv::BuiltInVertexIndex);
        interface.push_back(vertex_index);
    }

    std::vector<uint32_t> Emit() {
        using namespace ShaderRecompiler::IR;
        struct Output { uint32_t variable; uint32_t type; bool array; };
        std::vector<Output> outputs;
        uint32_t clip = 0u;
        for (const auto& field : layout.fields) {
            if (field.output.kind == StageOutputKind::ClipDistance)
                clip = std::max(clip, field.output.index + 1u);
        }
        uint32_t clip_var = 0u;
        for (const auto& field : layout.fields) {
            const auto kind = field.output.kind;
            if (kind == StageOutputKind::CullDistance) {
                outputs.push_back({0u, float_type, false});
                continue;
            }
            const bool distance = kind == StageOutputKind::ClipDistance;
            const auto type = distance ? float_type : kind == StageOutputKind::Layer ? uint_type : vec4_type;
            uint32_t variable = 0u;
            if (distance) {
                if (clip_var == 0u) {
                    builder.RequireCapability(spv::CapabilityClipDistance);
                    clip_var = DefineOutput(builder.Type(spv::OpTypeArray, float_type, Uint(clip)));
                    builder.AddAnnotation(spv::OpDecorate, clip_var, spv::DecorationBuiltIn,
                                          spv::BuiltInClipDistance);
                }
                variable = clip_var;
            } else {
                variable = DefineOutput(type);
                if (kind == StageOutputKind::Parameter) {
                    builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationLocation, field.output.location);
                } else {
                    if (kind == StageOutputKind::Layer) builder.RequireCapability(spv::CapabilityShaderLayer);
                    builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationBuiltIn,
                        kind == StageOutputKind::Layer ? spv::BuiltInLayer : spv::BuiltInPosition);
                }
            }
            outputs.push_back({variable, type, distance});
        }
        const auto main = builder.AllocateId();
        builder.AddEntryPoint(spv::ExecutionModelVertex, main, "main", interface);
        builder.AddFunction(spv::OpFunction, void_type, main, spv::FunctionControlMaskNone,
                            builder.Type(spv::OpTypeFunction, void_type));
        builder.AddFunction(spv::OpLabel, builder.AllocateId());
        const auto vertex = Result(spv::OpBitcast, uint_type, Result(spv::OpLoad, int_type, vertex_index));
        const auto triangle = Binary(spv::OpUDiv, vertex, Uint(3u));
        const auto group = Binary(spv::OpUDiv, triangle, Uint(layout.max_primitives));
        const auto primitive = Binary(spv::OpUMod, triangle, Uint(layout.max_primitives));
        const auto base = Binary(spv::OpIAdd, Uint(MeshEmulationLayout::HeaderDwords),
                                  Binary(spv::OpIMul, group, Uint(layout.GroupDwords())));
        const auto vertices = Read(base);
        const auto primitives = Read(Binary(spv::OpIAdd, base, Uint(1u)));
        const auto packed = Read(Binary(spv::OpIAdd, base, Binary(spv::OpIAdd, primitive, Uint(2u))));
        auto valid = Result(spv::OpULessThan, bool_type, primitive, primitives);
        valid = Result(spv::OpLogicalAnd, bool_type, valid,
                       Result(spv::OpIEqual, bool_type,
                              Binary(spv::OpBitwiseAnd, packed, Uint(0x80000000u)), Uint(0u)));
        for (uint32_t corner = 0; corner < 3u; ++corner) {
            const auto index = Result(spv::OpBitFieldUExtract, uint_type, packed, Uint(corner * 10u), Uint(9u));
            valid = Result(spv::OpLogicalAnd, bool_type, valid,
                           Result(spv::OpULessThan, bool_type, index, vertices));
        }
        // CullDistance rejects a whole triangle when all three vertices are negative for
        // one plane. Evaluate it here because Metal has no native cull-distance output.
        for (const auto& field : layout.fields) {
            if (field.output.kind != StageOutputKind::CullDistance) continue;
            auto negative = builder.Constant(spv::OpConstantTrue, bool_type);
            const auto zero = builder.Constant(spv::OpConstant, float_type, 0u);
            for (uint32_t corner = 0; corner < 3u; ++corner) {
                const auto candidate = Result(spv::OpBitFieldUExtract, uint_type, packed,
                                               Uint(corner * 10u), Uint(9u));
                const auto safe = Result(spv::OpSelect, uint_type, valid, candidate, Uint(0u));
                const auto offset = Binary(spv::OpIAdd, base,
                    Binary(spv::OpIAdd, Uint(layout.VerticesOffset() + field.offset),
                           Binary(spv::OpIMul, safe, Uint(layout.vertex_dwords))));
                const auto distance = Result(spv::OpBitcast, float_type, Read(offset));
                negative = Result(spv::OpLogicalAnd, bool_type, negative,
                                  Result(spv::OpFOrdLessThan, bool_type, distance, zero));
            }
            valid = Result(spv::OpLogicalAnd, bool_type, valid,
                           Result(spv::OpLogicalNot, bool_type, negative));
        }
        const auto corner = Binary(spv::OpUMod, vertex, Uint(3u));
        const auto index = Result(spv::OpBitFieldUExtract, uint_type, packed,
                                  Binary(spv::OpIMul, corner, Uint(10u)), Uint(9u));
        const auto layer_index = Result(spv::OpBitFieldUExtract, uint_type, packed,
                                        Uint(provoking * 10u), Uint(9u));
        const auto data_base = Binary(spv::OpIAdd, base, Uint(layout.VerticesOffset()));
        for (uint32_t f = 0; f < layout.fields.size(); ++f) {
            const auto& field = layout.fields[f];
            const auto& output = outputs[f];
            if (output.variable == 0u) continue;
            const auto selected = field.output.kind == StageOutputKind::Layer ? layer_index : index;
            const auto safe_index = Result(spv::OpSelect, uint_type, valid, selected, Uint(0u));
            const auto offset = Binary(spv::OpIAdd, data_base,
                Binary(spv::OpIAdd, Uint(field.offset), Binary(spv::OpIMul, safe_index, Uint(layout.vertex_dwords))));
            uint32_t value;
            if (output.type == vec4_type) {
                std::array<uint32_t, 4> components{};
                for (uint32_t c = 0; c < 4u; ++c)
                    components[c] = Result(spv::OpBitcast, float_type, Read(Binary(spv::OpIAdd, offset, Uint(c))));
                value = Result(spv::OpCompositeConstruct, vec4_type,
                    components[0], components[1], components[2], components[3]);
                if (field.output.kind == StageOutputKind::Position) {
                    const auto zero = builder.Constant(spv::OpConstant, float_type, 0u);
                    const auto one = builder.Constant(spv::OpConstant, float_type, 0x3f800000u);
                    const auto degenerate = builder.Constant(spv::OpConstantComposite, vec4_type, zero, zero, zero, one);
                    value = Result(spv::OpSelect, vec4_type, valid, value, degenerate);
                }
            } else {
                value = Read(offset);
                if (output.type != uint_type) value = Result(spv::OpBitcast, output.type, value);
            }
            auto pointer = output.variable;
            if (output.array) pointer = Result(spv::OpAccessChain,
                Pointer(spv::StorageClassOutput, output.type), pointer, Uint(field.output.index));
            builder.AddFunction(spv::OpStore, pointer, value);
        }
        builder.AddFunction(spv::OpReturn);
        builder.AddFunction(spv::OpFunctionEnd);
        return builder.Build();
    }

private:
    uint32_t Uint(uint32_t v) { return builder.Constant(spv::OpConstant, uint_type, v); }
    uint32_t Pointer(spv::StorageClass storage_class, uint32_t type) {
        return builder.Type(spv::OpTypePointer, storage_class, type);
    }
    template<class... Args> uint32_t Result(spv::Op op, uint32_t type, Args... args) {
        const auto result = builder.AllocateId();
        builder.AddFunction(op, type, result, args...);
        return result;
    }
    uint32_t Binary(spv::Op op, uint32_t a, uint32_t b) { return Result(op, uint_type, a, b); }
    uint32_t Read(uint32_t offset) {
        const auto pointer = Result(spv::OpAccessChain, Pointer(spv::StorageClassStorageBuffer, uint_type),
                                    storage, Uint(0u), offset);
        return Result(spv::OpLoad, uint_type, pointer);
    }
    uint32_t DefineOutput(uint32_t type) {
        const auto variable = builder.DefineGlobalVariable(Pointer(spv::StorageClassOutput, type),
                                                           spv::StorageClassOutput);
        interface.push_back(variable);
        return variable;
    }
    ShaderRecompiler::Spirv::Builder builder{0x00010500u};
    const MeshEmulationLayout& layout;
    uint32_t provoking;
    uint32_t void_type, bool_type, uint_type, int_type, float_type, vec4_type, storage, vertex_index;
    std::vector<uint32_t> interface;
};

} // namespace

std::vector<uint32_t> BuildMeshEmulationVertexShader(const MeshEmulationLayout& layout,
                                                    uint32_t provoking_vertex) {
    EXIT_IF(provoking_vertex >= 3u);
    return MeshVertexEmitter(layout, provoking_vertex).Emit();
}

} // namespace Libs::Graphics
