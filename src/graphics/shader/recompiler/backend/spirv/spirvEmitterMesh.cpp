#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInstructions.h"
#include "graphics/shader/meshEmulation.h"

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {
namespace {

uint32_t MeshArray(EmitterState& state, spv::StorageClass storage, uint32_t type, uint32_t count) {
	const auto array = state.builder.Type(spv::OpTypeArray, type, ConstantU32(state, count));
	return state.builder.DefineGlobalVariable(TypePointer(state, storage, array), storage);
}

uint32_t MeshElement(EmitterState& state, uint32_t variable, spv::StorageClass storage,
                     uint32_t type, uint32_t index) {
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypePointer(state, storage, type), pointer,
	                          variable, index);
	return pointer;
}

uint32_t MeshLoad(EmitterState& state, uint32_t variable, spv::StorageClass storage, uint32_t type,
                  uint32_t index) {
	const auto pointer = MeshElement(state, variable, storage, type, index);
	const auto value   = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, type, value, pointer);
	return value;
}

uint32_t MeshOutputType(EmitterState& state, IR::StageOutputKind kind) {
	switch (kind) {
		case IR::StageOutputKind::Layer: return TypeU32(state);
		case IR::StageOutputKind::ClipDistance:
		case IR::StageOutputKind::CullDistance: return TypeF32(state);
		case IR::StageOutputKind::Position:
		case IR::StageOutputKind::Parameter: return TypeF32Vector(state, 4);
		default: EXIT("unsupported mesh output kind=%u\n", static_cast<uint32_t>(kind));
	}
}

} // namespace

void DefineMeshOutputs(EmitterState& state, uint32_t clip_distance_count,
                       uint32_t cull_distance_count) {
	const auto& mesh = state.input_info.vertex->mesh;
	for (auto& output: state.outputs) {
		const auto type = MeshOutputType(state, output.kind);
		const bool clip = output.kind == IR::StageOutputKind::ClipDistance;
		const bool cull = output.kind == IR::StageOutputKind::CullDistance;
		auto& variable = clip ? state.clip_distance_variable
		                 : cull ? state.cull_distance_variable : output.variable_id;
		if (!state.MeshEmulated() && variable == 0) {
			const auto element_type = clip || cull
			                              ? state.builder.Type(spv::OpTypeArray, type,
			                                                   ConstantU32(state, clip ? clip_distance_count
			                                                                           : cull_distance_count))
			                              : type;
			variable = MeshArray(state, spv::StorageClassOutput, element_type,
			                     output.kind == IR::StageOutputKind::Layer ? mesh.max_primitives
			                                                             : mesh.max_vertices);
			state.interface_variables.push_back(variable);
			state.builder.AddName(variable, output.debug_name.c_str());
			if (output.kind == IR::StageOutputKind::Parameter) {
				state.builder.AddAnnotation(spv::OpDecorate, variable,
				                            spv::DecorationLocation, output.location);
			} else {
				const auto builtin = clip ? spv::BuiltInClipDistance
				                     : cull ? spv::BuiltInCullDistance
				                     : output.kind == IR::StageOutputKind::Layer ? spv::BuiltInLayer
				                                                                 : spv::BuiltInPosition;
				state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationBuiltIn, builtin);
			}
			if (output.kind == IR::StageOutputKind::Layer) {
				state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationPerPrimitiveEXT);
			}
		}
		output.variable_id = variable;
		// Only Layer is read by another invocation, through the primitive's provoking vertex.
		const bool shared = output.kind == IR::StageOutputKind::Layer;
		output.mesh_data_variable =
		    MeshArray(state, shared ? spv::StorageClassWorkgroup : spv::StorageClassPrivate, type,
		              shared ? mesh.max_vertices : state.lane_count);
	}
	state.mesh_allocation = MeshArray(state, spv::StorageClassWorkgroup, TypeU32(state), 2);
	state.mesh_primitive_data =
	    MeshArray(state, spv::StorageClassPrivate, TypeU32(state), state.lane_count);
	if (state.MeshEmulated()) return;
	state.mesh_primitives =
	    MeshArray(state, spv::StorageClassOutput, TypeU32Vector(state, 3), mesh.max_primitives);
	state.mesh_cull =
	    MeshArray(state, spv::StorageClassOutput, TypeBool(state), mesh.max_primitives);
	state.interface_variables.push_back(state.mesh_primitives);
	state.interface_variables.push_back(state.mesh_cull);
	state.builder.AddAnnotation(
	    spv::OpDecorate, state.mesh_primitives, spv::DecorationBuiltIn,
	    spv::BuiltInPrimitiveTriangleIndicesEXT); // PrimitiveTriangleIndicesEXT
	state.builder.AddAnnotation(spv::OpDecorate, state.mesh_cull, spv::DecorationBuiltIn,
	                            spv::BuiltInCullPrimitiveEXT); // CullPrimitiveEXT
	state.builder.AddAnnotation(spv::OpDecorate, state.mesh_cull, spv::DecorationPerPrimitiveEXT);
}

uint32_t MeshOutputPointer(EmitterState& state, IR::StageOutputKind kind, uint32_t index) {
	const auto output = std::ranges::find_if(state.outputs, [=](const OutputBinding& binding) {
		return binding.kind == kind && binding.index == index;
	});
	if (output == state.outputs.end()) {
		EXIT("mesh export has no output binding: kind=%u index=%u\n", static_cast<uint32_t>(kind),
		     index);
	}
	const bool shared = kind == IR::StageOutputKind::Layer;
	return MeshElement(
	    state, output->mesh_data_variable,
	    shared ? spv::StorageClassWorkgroup : spv::StorageClassPrivate, MeshOutputType(state, kind),
	    shared ? EmitLocalInvocationIndex(state) : ConstantU32(state, state.lane_half));
}

uint32_t MeshPrimitivePointer(EmitterState& state) {
	return MeshElement(state, state.mesh_primitive_data, spv::StorageClassPrivate, TypeU32(state),
	                   ConstantU32(state, state.lane_half));
}

void EmitMeshAllocate(ValueEmitContext& ctx, const IR::Inst& inst) {
	auto&      state = ctx.state;
	const auto first = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpIEqual, TypeBool(state), first,
	                          EmitLocalInvocationIndex(state), ConstantU32(state, 0));
	EmitIfCondition(state, first, [&] {
		const auto allocation = ctx.Arg(inst, 0);
		for (uint32_t field = 0; field < 2; field++) {
			const auto value = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpBitFieldUExtract, TypeU32(state), value, allocation,
			                          ConstantU32(state, field * 12u),
			                          ConstantU32(state, field == 0 ? 10u : 11u));
			const auto pointer =
			    MeshElement(state, state.mesh_allocation, spv::StorageClassWorkgroup,
			                TypeU32(state), ConstantU32(state, field));
			state.builder.AddFunction(spv::OpStore, pointer, value);
		}
	});
}

uint32_t EmitMeshEmulationHeaderLoad(EmitterState& state, uint32_t index) {
	const auto pointer = state.builder.AllocateId();
	const auto value = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain,
	    TypePointer(state, spv::StorageClassStorageBuffer, TypeU32(state)), pointer,
	    state.mesh_output_buffer, ConstantU32(state, 0u), ConstantU32(state, index));
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
	return value;
}

static void EmitEmulatedMeshOutputs(EmitterState& state, uint32_t vertices, uint32_t primitives) {
	const auto layout = BuildMeshEmulationLayout(state.program.info.outputs, *state.input_info.vertex);
	const auto Bound = [&](uint32_t count, uint32_t limit) {
		const auto fits = state.builder.AllocateId();
		const auto bounded = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpULessThan, TypeBool(state), fits, count, ConstantU32(state, limit));
		state.builder.AddFunction(spv::OpSelect, TypeU32(state), bounded, fits, count, ConstantU32(state, limit));
		return bounded;
	};
	vertices = Bound(vertices, layout.max_vertices);
	primitives = Bound(primitives, layout.max_primitives);
	const auto group = EmitAddU32(state,
	    EmitInputComponentU32(state, IR::StageInputKind::WorkgroupId, 0u),
	    EmitBinaryU32(state, spv::OpIMul,
	        EmitInputComponentU32(state, IR::StageInputKind::WorkgroupId, 1u),
	        EmitInputComponentU32(state, IR::StageInputKind::NumWorkgroups, 0u)));
	const auto base = EmitAddU32(state, ConstantU32(state, MeshEmulationLayout::HeaderDwords),
	    EmitBinaryU32(state, spv::OpIMul, group, ConstantU32(state, layout.GroupDwords())));
	const auto Store = [&](uint32_t offset, uint32_t value) {
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain,
		    TypePointer(state, spv::StorageClassStorageBuffer, TypeU32(state)), pointer,
		    state.mesh_output_buffer, ConstantU32(state, 0u), offset);
		state.builder.AddFunction(spv::OpStore, pointer, value);
	};
	for (uint32_t half = 0; half < state.lane_count; ++half) {
		state.lane_half = half;
		const auto index = EmitLocalInvocationIndex(state);
		const auto first = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpIEqual, TypeBool(state), first, index, ConstantU32(state, 0u));
		EmitIfCondition(state, first, [&] {
			Store(base, vertices);
			Store(EmitAddU32(state, base, ConstantU32(state, 1u)), primitives);
		});
		const auto is_vertex = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpULessThan, TypeBool(state), is_vertex, index, vertices);
		EmitIfCondition(state, is_vertex, [&] {
			const auto vertex_base = EmitAddU32(state, base,
			    EmitAddU32(state, ConstantU32(state, layout.VerticesOffset()),
			        EmitBinaryU32(state, spv::OpIMul, index, ConstantU32(state, layout.vertex_dwords))));
			EXIT_IF(layout.fields.size() != state.outputs.size());
			for (uint32_t f = 0; f < state.outputs.size(); ++f) {
				const auto& output = state.outputs[f];
				const bool shared = output.kind == IR::StageOutputKind::Layer;
				const bool vector = output.kind == IR::StageOutputKind::Position ||
				                    output.kind == IR::StageOutputKind::Parameter;
				auto value = MeshLoad(state, output.mesh_data_variable,
				    shared ? spv::StorageClassWorkgroup : spv::StorageClassPrivate,
				    MeshOutputType(state, output.kind), shared ? index : ConstantU32(state, half));
				if (!shared) {
					const auto bits = state.builder.AllocateId();
					state.builder.AddFunction(spv::OpBitcast, vector ? TypeU32Vector(state, 4u) : TypeU32(state), bits, value);
					value = bits;
				}
				for (uint32_t c = 0; c < (vector ? 4u : 1u); ++c) {
					auto component = value;
					if (vector) {
						component = state.builder.AllocateId();
						state.builder.AddFunction(spv::OpCompositeExtract, TypeU32(state), component, value, c);
					}
					Store(EmitAddU32(state, vertex_base, ConstantU32(state, layout.fields[f].offset + c)), component);
				}
			}
		});
		const auto is_primitive = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpULessThan, TypeBool(state), is_primitive, index, primitives);
		EmitIfCondition(state, is_primitive, [&] {
			Store(EmitAddU32(state, base, EmitAddU32(state, ConstantU32(state, 2u), index)),
			    MeshLoad(state, state.mesh_primitive_data, spv::StorageClassPrivate,
			             TypeU32(state), ConstantU32(state, half)));
		});
	}
	state.lane_half = 0u;
}

void EmitMeshEntryPoint(EmitterState& state) {
	state.builder.AddFunction(spv::OpFunction, TypeVoid(state), state.main_func,
	                          spv::FunctionControlMaskNone, TypeFunction(state));
	EmitLabel(state, state.builder.AllocateId());
	state.builder.AddFunction(spv::OpFunctionCall, TypeVoid(state), state.builder.AllocateId(),
	                          state.mesh_guest_func);
	// All guest waves finish before the uniform Vulkan allocation and output stores.
	EmitBarrier(state);
	const auto vertices   = MeshLoad(state, state.mesh_allocation, spv::StorageClassWorkgroup,
	                                 TypeU32(state), ConstantU32(state, 0));
	const auto primitives = MeshLoad(state, state.mesh_allocation, spv::StorageClassWorkgroup,
	                                 TypeU32(state), ConstantU32(state, 1));
	if (state.MeshEmulated()) {
		EmitEmulatedMeshOutputs(state, vertices, primitives);
		state.builder.AddFunction(spv::OpReturn);
		state.builder.AddFunction(spv::OpFunctionEnd);
		return;
	}
	state.builder.AddFunction(spv::OpSetMeshOutputsEXT, vertices,
	                          primitives); // OpSetMeshOutputsEXT
	for (uint32_t half = 0; half < state.lane_count; half++) {
		state.lane_half      = half;
		const auto index     = EmitLocalInvocationIndex(state);
		const auto is_vertex = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpULessThan, TypeBool(state), is_vertex, index, vertices);
		EmitIfCondition(state, is_vertex, [&] {
			for (const auto& output: state.outputs) {
				if (output.kind == IR::StageOutputKind::Layer) {
					continue;
				}
				const auto type  = MeshOutputType(state, output.kind);
				const auto value =
				    MeshLoad(state, output.mesh_data_variable, spv::StorageClassPrivate, type,
				             ConstantU32(state, half));
				uint32_t pointer;
				if (output.kind == IR::StageOutputKind::ClipDistance ||
				    output.kind == IR::StageOutputKind::CullDistance) {
					pointer = state.builder.AllocateId();
					state.builder.AddFunction(
					    spv::OpAccessChain, TypePointer(state, spv::StorageClassOutput, type), pointer,
					    output.variable_id, index, ConstantU32(state, output.index));
				} else {
					pointer = MeshElement(state, output.variable_id, spv::StorageClassOutput, type, index);
				}
				state.builder.AddFunction(spv::OpStore, pointer, value);
			}
		});
		const auto is_primitive = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpULessThan, TypeBool(state), is_primitive, index,
		                          primitives);
		EmitIfCondition(state, is_primitive, [&] {
			const auto packed = MeshLoad(state, state.mesh_primitive_data, spv::StorageClassPrivate,
			                             TypeU32(state), ConstantU32(state, half));
			uint32_t   vertex[3] {};
			for (uint32_t component = 0; component < 3; component++) {
				vertex[component] = state.builder.AllocateId();
				state.builder.AddFunction(
				    spv::OpBitFieldUExtract, TypeU32(state), vertex[component], packed,
				    ConstantU32(state, component * 10u), ConstantU32(state, 9));
			}
			const auto triangle = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 3), triangle,
			                          vertex[0], vertex[1], vertex[2]);
			const auto triangle_pointer =
			    MeshElement(state, state.mesh_primitives, spv::StorageClassOutput,
			                TypeU32Vector(state, 3), index);
			state.builder.AddFunction(spv::OpStore, triangle_pointer, triangle);
			const auto null_bit =
			    EmitBinaryU32(state, spv::OpBitwiseAnd, packed, ConstantU32(state, 0x80000000u));
			const auto culled = state.builder.AllocateId();
			state.builder.AddFunction(spv::OpINotEqual, TypeBool(state), culled, null_bit,
			                          ConstantU32(state, 0));
			state.builder.AddFunction(spv::OpStore,
			                          MeshElement(state, state.mesh_cull, spv::StorageClassOutput,
			                                      TypeBool(state), index),
			                          culled);
			for (const auto& output: state.outputs) {
				if (output.kind != IR::StageOutputKind::Layer) {
					continue;
				}
				const auto layer = MeshLoad(state, output.mesh_data_variable,
				                            spv::StorageClassWorkgroup, TypeU32(state),
				                            vertex[state.input_info.vertex->mesh.provoking_vertex]);
				const auto pointer = MeshElement(state, output.variable_id, spv::StorageClassOutput,
				                                 TypeU32(state), index);
				state.builder.AddFunction(spv::OpStore, pointer, layer);
			}
		});
	}
	state.lane_half = 0;
	state.builder.AddFunction(spv::OpReturn);
	state.builder.AddFunction(spv::OpFunctionEnd);
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
