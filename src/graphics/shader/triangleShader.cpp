#include "graphics/shader/triangleShader.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvBuilder.h"

#include <algorithm>
#include <array>

namespace Libs::Graphics {

std::vector<TriangleParameter> TriangleParameters(
    const ShaderPixelInputInfo& pixel, const ShaderRecompiler::IR::ShaderInfo& info) {
    using namespace ShaderRecompiler::IR;
    if (!pixel.emulate_per_vertex || std::ranges::none_of(info.inputs, [](const auto& input) {
            return (input.kind == StageInputKind::Parameter && input.per_vertex) ||
                   input.kind == StageInputKind::BaryCoordSmooth ||
                   input.kind == StageInputKind::BaryCoordNoPerspective;
        })) return {};
    std::vector<TriangleParameter> parameters;
    uint32_t location = 0;
    for (const auto& input : info.inputs) {
        if (input.kind != StageInputKind::Parameter) continue;
        const auto mapped = ShaderPixelParameterMappedLocation(pixel, input.location);
        if (std::ranges::any_of(parameters, [&](const auto& p) { return p.input_location == mapped; }))
            continue;
        parameters.push_back({mapped, location, input.per_vertex,
                              ShaderPixelParameterIsFlat(pixel, input.location),
                              pixel.ps_no_perspective});
        location += input.per_vertex ? 3u : 1u;
    }
    for (const auto& input : info.inputs) {
        if (input.kind != StageInputKind::BaryCoordSmooth &&
            input.kind != StageInputKind::BaryCoordNoPerspective) continue;
        parameters.push_back({.output_location = location++,
                              .no_perspective = input.kind == StageInputKind::BaryCoordNoPerspective,
                              .source = input.kind});
    }
    EXIT_IF(location > 32u);
    return parameters;
}

std::vector<uint32_t> ExpandTriangleIndices(Prospero::PrimitiveType primitive,
                                          std::span<const uint32_t> indices,
                                          bool restart, uint32_t restart_index, bool provoking_last) {
    EXIT_IF(primitive != Prospero::PrimitiveType::kTriStrip &&
            primitive != Prospero::PrimitiveType::kTriFan);
    std::vector<uint32_t> triangles;
    if (indices.size() > 2u) triangles.reserve((indices.size() - 2u) * 3u);
    uint32_t first = 0, previous = 0, count = 0;
    for (const auto index : indices) {
        if (restart && index == restart_index) { count = 0; continue; }
        if (count == 0u) first = index;
        else if (count >= 2u) {
            if (primitive == Prospero::PrimitiveType::kTriFan) {
                triangles.insert(triangles.end(), provoking_last
                    ? std::initializer_list<uint32_t>{first, previous, index}
                    : std::initializer_list<uint32_t>{previous, index, first});
            } else {
                const bool reverse = (count & 1u) != 0;
                if (reverse && provoking_last)
                    triangles.insert(triangles.end(), {previous, first, index});
                else
                    triangles.insert(triangles.end(), {first, reverse ? index : previous,
                                                       reverse ? previous : index});
                first = previous;
            }
        }
        previous = index;
        ++count;
    }
    return triangles;
}

namespace {

class TriangleEmitter {
public:
    TriangleEmitter(std::span<const TriangleParameter> parameters_, uint32_t clips_,
                    uint32_t provoking_, bool control_)
        : parameters(parameters_), clips(clips_), provoking(provoking_), control(control_) {
        builder.RequireCapability(spv::CapabilityShader);
        builder.RequireCapability(spv::CapabilityTessellation);
        if (clips) builder.RequireCapability(spv::CapabilityClipDistance);
        builder.AddMemoryModel(spv::AddressingModelLogical, spv::MemoryModelGLSL450);
        void_type = builder.Type(spv::OpTypeVoid);
        bool_type = builder.Type(spv::OpTypeBool);
        uint_type = builder.Type(spv::OpTypeInt, 32u, 0u);
        int_type = builder.Type(spv::OpTypeInt, 32u, 1u);
        float_type = builder.Type(spv::OpTypeFloat, 32u);
        vec3_type = builder.Type(spv::OpTypeVector, float_type, 3u);
        vec4_type = builder.Type(spv::OpTypeVector, float_type, 4u);
        if (clips) {
            vertex_type = builder.DecoratedType(spv::OpTypeStruct,
                {{spv::OpDecorate, {spv::DecorationBlock}},
                 {spv::OpMemberDecorate, {0u, spv::DecorationBuiltIn, spv::BuiltInPosition}},
                 {spv::OpMemberDecorate, {1u, spv::DecorationBuiltIn, spv::BuiltInClipDistance}}},
                vec4_type, Array(float_type, clips));
        } else {
            vertex_type = builder.DecoratedType(spv::OpTypeStruct,
                {{spv::OpDecorate, {spv::DecorationBlock}},
                 {spv::OpMemberDecorate, {0u, spv::DecorationBuiltIn, spv::BuiltInPosition}}}, vec4_type);
        }
    }

    std::vector<uint32_t> Emit() {
        const auto in_vertex = Interface(spv::StorageClassInput, Array(vertex_type, 3u));
        const auto out_vertex = Interface(spv::StorageClassOutput,
                                         control ? Array(vertex_type, 3u) : vertex_type);
        const auto selector = Interface(spv::StorageClassInput, control ? int_type : vec3_type);
        Decorate(selector, spv::DecorationBuiltIn,
                 control ? spv::BuiltInInvocationId : spv::BuiltInTessCoord);
        std::vector<uint32_t> inputs, outputs;
        for (const auto& parameter : parameters) {
            if (parameter.source != ShaderRecompiler::IR::StageInputKind::Parameter) {
                // Barycentric basis values are generated at the original three corners.
                inputs.push_back(0u);
                outputs.push_back(control ? 0u : Interface(spv::StorageClassOutput, vec3_type));
                if (!control) {
                    Decorate(outputs.back(), spv::DecorationLocation, parameter.output_location);
                    if (parameter.no_perspective)
                        builder.AddAnnotation(spv::OpDecorate, outputs.back(), spv::DecorationNoPerspective);
                }
                continue;
            }
            inputs.push_back(Interface(spv::StorageClassInput, Array(vec4_type, 3u)));
            Decorate(inputs.back(), spv::DecorationLocation, parameter.input_location);
            outputs.push_back(Interface(spv::StorageClassOutput,
                control || parameter.per_vertex ? Array(vec4_type, 3u) : vec4_type));
            Decorate(outputs.back(), spv::DecorationLocation,
                     control ? parameter.input_location : parameter.output_location);
            if (!control) {
                if (parameter.per_vertex || parameter.flat)
                    builder.AddAnnotation(spv::OpDecorate, outputs.back(), spv::DecorationFlat);
                else if (parameter.no_perspective)
                    builder.AddAnnotation(spv::OpDecorate, outputs.back(), spv::DecorationNoPerspective);
            }
        }
        uint32_t inner = 0, outer = 0;
        if (control) {
            inner = Interface(spv::StorageClassOutput, Array(float_type, 2u));
            outer = Interface(spv::StorageClassOutput, Array(float_type, 4u));
            Decorate(inner, spv::DecorationBuiltIn, spv::BuiltInTessLevelInner);
            Decorate(outer, spv::DecorationBuiltIn, spv::BuiltInTessLevelOuter);
            builder.AddAnnotation(spv::OpDecorate, inner, spv::DecorationPatch);
            builder.AddAnnotation(spv::OpDecorate, outer, spv::DecorationPatch);
        }
        const auto main = builder.AllocateId();
        builder.AddEntryPoint(control ? spv::ExecutionModelTessellationControl
                                      : spv::ExecutionModelTessellationEvaluation,
                              main, "main", interfaces);
        if (control) builder.AddExecutionMode(main, spv::ExecutionModeOutputVertices, 3u);
        else {
            builder.AddExecutionMode(main, spv::ExecutionModeTriangles);
            builder.AddExecutionMode(main, spv::ExecutionModeSpacingEqual);
            builder.AddExecutionMode(main, spv::ExecutionModeVertexOrderCw);
        }
        builder.AddFunction(spv::OpFunction, void_type, main, spv::FunctionControlMaskNone,
                            builder.Type(spv::OpTypeFunction, void_type));
        builder.AddFunction(spv::OpLabel, builder.AllocateId());
        uint32_t index;
        if (control) {
            index = Result(spv::OpBitcast, uint_type, Result(spv::OpLoad, int_type, selector));
            const auto first = Result(spv::OpIEqual, bool_type, index, Uint(0u));
            const auto write = builder.AllocateId(), merge = builder.AllocateId();
            builder.AddFunction(spv::OpSelectionMerge, merge, spv::SelectionControlMaskNone);
            builder.AddFunction(spv::OpBranchConditional, first, write, merge);
            builder.AddFunction(spv::OpLabel, write);
            const auto one = builder.Constant(spv::OpConstant, float_type, 0x3f800000u);
            for (uint32_t i = 0; i < 2u; ++i) StoreAt(inner, float_type, one, Uint(i));
            for (uint32_t i = 0; i < 4u; ++i) StoreAt(outer, float_type, one, Uint(i));
            builder.AddFunction(spv::OpBranch, merge);
            builder.AddFunction(spv::OpLabel, merge);
        } else {
            const auto coordinates = Result(spv::OpLoad, vec3_type, selector);
            const auto zero = builder.Constant(spv::OpConstant, float_type, 0u);
            const auto x = Result(spv::OpCompositeExtract, float_type, coordinates, 0u);
            const auto y = Result(spv::OpCompositeExtract, float_type, coordinates, 1u);
            index = Result(spv::OpSelect, uint_type, Result(spv::OpFOrdGreaterThan, bool_type, y, zero),
                           Uint(1u), Uint(2u));
            index = Result(spv::OpSelect, uint_type, Result(spv::OpFOrdGreaterThan, bool_type, x, zero),
                           Uint(0u), index);
        }
        const auto position = LoadAt(in_vertex, vec4_type, index, Uint(0u));
        if (control) StoreAt(out_vertex, vec4_type, position, index, Uint(0u));
        else StoreAt(out_vertex, vec4_type, position, Uint(0u));
        for (uint32_t i = 0; i < clips; ++i) {
            const auto value = LoadAt(in_vertex, float_type, index, Uint(1u), Uint(i));
            if (control) StoreAt(out_vertex, float_type, value, index, Uint(1u), Uint(i));
            else StoreAt(out_vertex, float_type, value, Uint(1u), Uint(i));
        }
        for (uint32_t i = 0; i < parameters.size(); ++i) {
            const auto& parameter = parameters[i];
            if (parameter.source != ShaderRecompiler::IR::StageInputKind::Parameter) {
                if (!control) {
                    const auto zero = builder.Constant(spv::OpConstant, float_type, 0u);
                    const auto one = builder.Constant(spv::OpConstant, float_type, 0x3f800000u);
                    std::array<uint32_t, 3> basis{};
                    for (uint32_t corner = 0; corner < 3u; ++corner)
                        basis[corner] = Result(spv::OpSelect, float_type,
                            Result(spv::OpIEqual, bool_type, index, Uint(corner)), one, zero);
                    builder.AddFunction(spv::OpStore, outputs[i], Result(spv::OpCompositeConstruct,
                        vec3_type, basis[0], basis[1], basis[2]));
                }
            } else if (control) {
                StoreAt(outputs[i], vec4_type, LoadAt(inputs[i], vec4_type, index), index);
            } else if (parameter.per_vertex) {
                for (uint32_t corner = 0; corner < 3u; ++corner)
                    StoreAt(outputs[i], vec4_type, LoadAt(inputs[i], vec4_type, Uint(corner)), Uint(corner));
            } else {
                const auto vertex = parameter.flat ? Uint(provoking) : index;
                builder.AddFunction(spv::OpStore, outputs[i], LoadAt(inputs[i], vec4_type, vertex));
            }
        }
        builder.AddFunction(spv::OpReturn);
        builder.AddFunction(spv::OpFunctionEnd);
        return builder.Build();
    }

private:
    template<class... Args> uint32_t Result(spv::Op op, uint32_t type, Args... args) {
        const auto id = builder.AllocateId();
        builder.AddFunction(op, type, id, args...);
        return id;
    }
    uint32_t Uint(uint32_t v) { return builder.Constant(spv::OpConstant, uint_type, v); }
    uint32_t Array(uint32_t type, uint32_t count) { return builder.Type(spv::OpTypeArray, type, Uint(count)); }
    uint32_t Pointer(spv::StorageClass storage, uint32_t type) {
        return builder.Type(spv::OpTypePointer, storage, type);
    }
    uint32_t Interface(spv::StorageClass storage, uint32_t type) {
        const auto id = builder.DefineGlobalVariable(Pointer(storage, type), storage);
        interfaces.push_back(id);
        return id;
    }
    void Decorate(uint32_t id, spv::Decoration decoration, uint32_t value) {
        builder.AddAnnotation(spv::OpDecorate, id, decoration, value);
    }
    template<class... Args> uint32_t LoadAt(uint32_t base, uint32_t type, Args... indices) {
        return Result(spv::OpLoad, type, Result(spv::OpAccessChain,
            Pointer(spv::StorageClassInput, type), base, indices...));
    }
    template<class... Args> void StoreAt(uint32_t base, uint32_t type, uint32_t value, Args... indices) {
        builder.AddFunction(spv::OpStore, Result(spv::OpAccessChain,
            Pointer(spv::StorageClassOutput, type), base, indices...), value);
    }
    ShaderRecompiler::Spirv::Builder builder{0x00010500u};
    std::span<const TriangleParameter> parameters;
    uint32_t clips, provoking;
    bool control;
    uint32_t void_type, bool_type, uint_type, int_type, float_type, vec3_type, vec4_type, vertex_type;
    std::vector<uint32_t> interfaces;
};

} // namespace

RectListShaders BuildTriangleShaders(std::span<const TriangleParameter> parameters,
                                    uint32_t clip_distances, uint32_t provoking_vertex) {
    EXIT_IF(provoking_vertex >= 3u);
    return {TriangleEmitter(parameters, clip_distances, provoking_vertex, true).Emit(),
            TriangleEmitter(parameters, clip_distances, provoking_vertex, false).Emit()};
}

} // namespace Libs::Graphics
