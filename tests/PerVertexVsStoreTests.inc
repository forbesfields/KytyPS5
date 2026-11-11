TestCompileResult RecompileForVertexStoreTest(std::span<const uint32_t> code,
    const ShaderRecompiler::CompileOptions& options,
    const ShaderRecompiler::Spirv::PvStoreInfo& store) {
  auto result = RecompileForTest(code, options);
  result.spirv = ShaderRecompiler::Spirv::EmitProgram(result.program, options.input_info, 0,
    *options.profile, options.emit_debug_info, store);
  return result;
}

// The consumer uses flat DWORD addresses, not float varyings for the raw corners.
// The compute producer and its barrier are the same as the old mesh path.
void TestPerVertexBufferEmission() {
  using namespace ShaderRecompiler;
  const uint32_t code[]{EncodeVintrp(2, 12, 0, 0, 2), EncodeVintrp(2, 13, 0, 0, 0),
    EncodeVintrp(2, 14, 0, 0, 1), EncodeExp0(0x00, 0xf), EncodeExp1(12, 13, 14, 12),
    EncodeSopp(0x01)};
  ShaderPixelInputInfo pixel{};
  pixel.input_num = 1;
  pixel.custom_interpolation_mask = 1;
  pixel.emulate_per_vertex = true;
  pixel.target_output_mode[0] = 4;
  auto options = MakeCompileOptions(ShaderType::Pixel);
  options.input_info.pixel = &pixel;
  const auto old = RecompileForTest(code, options);
  const Spirv::PvStoreInfo profile{.per_vertex_buffer = true};
  const auto result = RecompileForVertexStoreTest(code, options, profile);
  CheckSpirvBinaryValidates(result.spirv);
  const auto text = DisassembleSpirvBinary(result.spirv);
  Check(text.find("Binding 255") != std::string::npos &&
        text.find("OpTypeVector %uint 3") != std::string::npos,
        "buffer corner PS lost its storage buffer or integer corner addresses");
  Check(text.find("PerVertexKHR") == std::string::npos &&
        !SpirvContainsCapability(result.spirv, 5284u),
        "buffer corner PS requires native fragment barycentrics");
  const auto parameters = TriangleParameters(pixel, result.program.info);
  ShaderVertexInputInfo input{};
  input.mesh.max_vertices = 3;
  input.mesh.max_primitives = 1;
  const std::vector<IR::StageOutput> fields{
    {IR::StageOutputKind::Position, 0, 0, "position"},
    {IR::StageOutputKind::Parameter, 0, 0, "raw"}};
  const auto layout = BuildMeshEmulationLayout(fields, input);
  const auto vertex = BuildMeshEmulationVertexShader(layout, 0, parameters);
  CheckSpirvBinaryValidates(vertex);
  Check(!SpirvContainsCapability(vertex, 3u), "buffer corner VS still uses tessellation");
  Check(RecompileForTest(code, options).spirv == old.spirv,
        "buffer corner emission changed the fallback module");
  const uint32_t vertex_code[]{EncodeExp0(0x0c, 0xf), EncodeExp1(0,1,2,3),
    EncodeExp0(0x20, 0xf), EncodeExp1(4,5,6,7), EncodeSopp(0x01)};
  auto capture_options = MakeCompileOptions(ShaderType::Vertex);
  const Spirv::PvStoreInfo capture_profile{.vertex_store = true};
  const auto captured = RecompileForVertexStoreTest(vertex_code, capture_options, capture_profile);
  CheckSpirvBinaryValidates(captured.spirv);
  const auto capture_text = DisassembleSpirvBinary(captured.spirv);
  Check(capture_text.find("OpAtomicExchange") != std::string::npos &&
        capture_text.find("DescriptorSet 2") != std::string::npos,
        "vertex capture lost atomic records or its private descriptor set");
  const auto capture_layout = BuildVertexStoreLayout(captured.program.info.outputs);
  CheckSpirvBinaryValidates(BuildVertexStoreVertexShader(capture_layout, 0, parameters));
  const std::array<uint32_t, 3> indices{10, 12, 11};
  const auto plan = PlanVertexStore(indices, -8, 2, 9, 1024);
  Check(plan.valid && plan.first == 2 && plan.span == 3 && plan.records_offset == 11 &&
        plan.bytes == (11 + 6*9)*4 && plan.indices == std::vector<uint32_t>({0,2,1}),
        "vertex capture range or instance record math differs from upstream range semantics");
  Check(!PlanVertexStore(indices, -11, 1, 9, 1024).valid,
        "vertex capture accepts a negative effective vertex index");
  Check(!PlanVertexStore(indices, 0, UINT32_MAX, 9, UINT64_MAX).valid,
        "vertex capture overflows its shader DWORD offsets");
  Check(!PlanVertexStore(indices, 0, 2, 9, 32).valid,
        "vertex capture exceeds its storage-buffer bound");
  const uint32_t half_code[]{
    EncodeVop3Word0(0x342, 21), EncodeVop3Word1(0x1c0, 256, 0),
    EncodeVop3Word0(0x35a, 14), EncodeVop3Word1(0x1c0, 257, 277),
    EncodeVop3Word0(0x342, 15), EncodeVop3Word1(0x080, 256, 0),
    EncodeVop3Word0(0x35a, 14, 8), EncodeVop3Word1(0x080, 257, 271),
    EncodeExp0(0x00, 0x1), EncodeExp1(14,0,0,0), EncodeSopp(0x01)};
  pixel.custom_interpolation_mask = 0;
  const auto half = RecompileForVertexStoreTest(half_code, options, profile);
  CheckSpirvBinaryValidates(half.spirv);
  Check(SpirvExtInstCount(half.spirv, 62u) == 6u,
        "buffer corner FP16 input does not unpack both halves at all three corners");
  TestTriangleIndexExpansion();
  std::puts("VS-store CPU: buffer interface, capture, FP16, bounded ranges, fallback, strip/fan passed");
}

void TestPerVertexBufferGpu() {
  using namespace ShaderRecompiler;
  MeshGpuTest gpu;
  Spirv::PvStoreInfo profile{.per_vertex_buffer = true};
  ShaderVertexInputInfo input{};
  input.mesh.max_vertices = 3;
  input.mesh.max_primitives = 1;
  const std::vector<IR::StageOutput> fields{
    {IR::StageOutputKind::Position, 0, 0, "position"},
    {IR::StageOutputKind::Parameter, 0, 0, "raw"},
    {IR::StageOutputKind::CullDistance, 0, 0, "cull"}};
  const auto layout = BuildMeshEmulationLayout(fields, input);
  const std::array<float, 12> positions{-1,-1,0,1, 3,-1,0,1, -1,3,0,1};
  const auto record_for = [&](std::array<float, 3> w, std::array<uint32_t, 3> words,
                              bool visible = true) {
    std::vector<uint32_t> record(4 + layout.GroupDwords() + 16, 0xcdcdcdcdu);
    record[0] = record[1] = 0;
    record[2] = record[3] = 1;
    record[4] = 3;
    record[5] = 1;
    record[6] = 0x00200400u;
    for (uint32_t v = 0; v < 3; ++v) {
      const auto base = 4 + layout.VerticesOffset() + v * layout.vertex_dwords;
      for (uint32_t c = 0; c < 4; ++c) {
        record[base + c] = std::bit_cast<uint32_t>(positions[v*4+c] * w[v]);
        record[base + 4 + c] = words[v];
      }
      record[base + 8] = std::bit_cast<uint32_t>(visible && v == 2 ? 1.f : -1.f);
    }
    return record;
  };
  uint32_t cases = 0;
  // Compare every raw corner word, including NaN payloads, in the real guest PS.
  const std::array<uint32_t, 3> packed{0x7fc01234u, 0xff801234u, 0x12345678u};
  std::vector<uint32_t> code{EncodeVop1(0x01, 10, 242)};
  for (uint32_t corner = 0; corner < 3; ++corner) {
    const std::array<uint32_t, 5> part{
      EncodeVintrp(2, 12, 0, 0, (corner + 2u) % 3u),
      EncodeVop1(0x01, 20, 255), packed[corner],
      EncodeVopc(0xc2, 268, 20), EncodeVop2(0x01, 10, 128, 10)};
    code.insert(code.end(), part.begin(), part.end());
  }
  const std::array<uint32_t, 5> end{EncodeVop1(0x01, 0, 128), EncodeVop1(0x01, 1, 242),
    EncodeExp0(0x00, 0xf), EncodeExp1(0, 10, 0, 1), EncodeSopp(0x01)};
  code.insert(code.end(), end.begin(), end.end());
  ShaderPixelInputInfo pixel{};
  pixel.input_num = 1;
  pixel.custom_interpolation_mask = 1;
  pixel.emulate_per_vertex = true;
  pixel.target_output_mode[0] = 4;
  auto options = MakeCompileOptions(ShaderType::Pixel);
  options.input_info.pixel = &pixel;
  const auto raw = RecompileForVertexStoreTest(code, options, profile);
  const auto parameters = TriangleParameters(pixel, raw.program.info);
  for (const bool visible : {false, true}) {
    const auto record = record_for({1,1,1}, packed, visible);
    gpu.Render({}, layout, BuildMeshEmulationVertexShader(layout, 0, parameters), raw.spirv,
      record, {0, uint8_t(visible ? 255 : 0), 0, 255});
    ++cases;
  }
  // The second group must use its own record base. The first group is wholly culled.
  {
    auto groups = record_for({1,1,1}, packed, false);
    const auto second = record_for({1,1,1}, packed);
    groups.resize(4u + 2u * layout.GroupDwords() + 16u, 0xcdcdcdcdu);
    groups[2] = 2u;
    std::copy_n(second.begin() + 4u, layout.GroupDwords(),
      groups.begin() + 4u + layout.GroupDwords());
    gpu.Render({}, layout, BuildMeshEmulationVertexShader(layout, 0, parameters), raw.spirv,
      groups, {0,255,0,255}, nullptr, {}, {}, nullptr, 6u);
    ++cases;
  }
  // Flat P0 chooses the first or the last original corner, independent of raster provoking mode.
  const uint32_t flat_code[]{EncodeVintrp(2, 0, 0, 0, 2), EncodeVintrp(2, 14, 1, 0, 2),
    EncodeVop1(0x01, 1, 128), EncodeVop1(0x01, 2, 242),
    EncodeExp0(0x00, 0xf), EncodeExp1(0, 14, 1, 2), EncodeSopp(0x01)};
  const std::array<uint32_t, 3> values{std::bit_cast<uint32_t>(0.25f),
    std::bit_cast<uint32_t>(0.5f), std::bit_cast<uint32_t>(0.75f)};
  for (const bool last : {false, true}) {
    pixel.input_num = 2;
    pixel.custom_interpolation_mask = 2u; // The raw alias forces both inputs to retain their corners.
    pixel.parameter_mode = last ? ShaderPixelParameterMode::LastVertex : ShaderPixelParameterMode::FirstVertex;
    pixel.interpolator_settings[0] = 0x400u; // flat
    const auto flat = RecompileForVertexStoreTest(flat_code, options, profile);
    const auto params = TriangleParameters(pixel, flat.program.info);
    gpu.Render({}, layout, BuildMeshEmulationVertexShader(layout, last ? 2u : 0u, params), flat.spirv,
      record_for({1,1,1}, values), {uint8_t(last ? 191 : 64), 64, 0, 255});
    ++cases;
  }
  // Use both actual barycentric inputs. No native BaryCoordKHR is available on MoltenVK.
  const uint32_t bary_code[]{EncodeVop1(0x01, 2, 128), EncodeVop1(0x01, 3, 242),
    EncodeExp0(0x00, 0xf), EncodeExp1(0, 1, 2, 3), EncodeSopp(0x01)};
  for (const bool linear : {false, true}) {
    ShaderPixelInputInfo bary{};
    bary.emulate_per_vertex = true;
    bary.ps_no_perspective = linear;
    bary.ps_perspective_center_vgpr = 0;
    bary.ps_system_input_base = 2;
    bary.target_output_mode[0] = 4;
    options.input_info.pixel = &bary;
    auto fragment = RecompileForTest(bary_code, options);
    if (linear) {
      // The guest entry ABI exposes perspective-center VGPRs only. Select the linear builtin
      // in this emitter fixture; the guest parameter test below covers ps_no_perspective.
      for (auto& input : fragment.program.info.inputs)
        if (input.kind == IR::StageInputKind::BaryCoordSmooth)
          input.kind = IR::StageInputKind::BaryCoordNoPerspective;
      for (auto* block : fragment.program.blocks)
        for (auto& inst : *block)
          if (inst.GetOpcode() == IR::ValueOpcode::GetBuiltin &&
              inst.Arg(0).U32() == uint32_t(IR::StageInputKind::BaryCoordSmooth))
            inst.SetArg(0, IR::Value{uint32_t(IR::StageInputKind::BaryCoordNoPerspective)});
    }
    fragment.spirv = Spirv::EmitProgram(fragment.program, options.input_info, 0,
      *options.profile, options.emit_debug_info, profile);
    const auto params = TriangleParameters(bary, fragment.program.info);
    const auto vertex = BuildMeshEmulationVertexShader(layout, 0, params);
    for (const bool unequal : {false, true}) {
      const std::array<float, 3> w{1.f, unequal ? 2.f : 1.f, unequal ? 4.f : 1.f};
      std::array<std::array<uint8_t, 4>, 16> reference{};
      for (uint32_t y = 0; y < 4; ++y) {
        for (uint32_t x = 0; x < 4; ++x) {
          const float b = (float(x) + 0.5f) / 8.f, c = (float(y) + 0.5f) / 8.f;
          const float denominator = (1.f-b-c)/w[0] + b/w[1] + c/w[2];
          reference[y*4+x] = {uint8_t((linear ? b : b/w[1]/denominator)*255.f+0.5f),
            uint8_t((linear ? c : c/w[2]/denominator)*255.f+0.5f), 0, 255};
        }
      }
      gpu.Render({}, layout, vertex, fragment.spirv, record_for(w, values), {}, nullptr, {}, reference);
      ++cases;
    }
  }
  // Read the raw corners and interpolate a guest parameter in the same PS. Compare both
  // transport paths with a literal reference image, not only with each other.
  const uint32_t interpolated_code[]{EncodeVintrp(0, 12, 0, 0, 0), EncodeVintrp(1, 12, 0, 0, 0),
    EncodeVintrp(2, 13, 0, 0, 2), EncodeVop1(0x01, 2, 128), EncodeVop1(0x01, 3, 242),
    EncodeExp0(0x00, 0xf), EncodeExp1(12,13,2,3), EncodeSopp(0x01)};
  for (const bool linear : {false, true}) {
    ShaderPixelInputInfo smooth{};
    smooth.input_num = 1;
    smooth.emulate_per_vertex = true;
    smooth.ps_no_perspective = linear;
    smooth.target_output_mode[0] = 4;
    options.input_info.pixel = &smooth;
    const auto old = RecompileForTest(interpolated_code, options);
    const auto updated = RecompileForVertexStoreTest(interpolated_code, options, profile);
    const auto params = TriangleParameters(smooth, updated.program.info);
    const auto tess = BuildTriangleShaders(params, 0, 0);
    for (const bool unequal : {false, true}) {
      const std::array<float, 3> w{1.f, unequal ? 2.f : 1.f, unequal ? 4.f : 1.f};
      std::array<std::array<uint8_t, 4>, 16> reference{};
      for (uint32_t y = 0; y < 4; ++y) {
        for (uint32_t x = 0; x < 4; ++x) {
          const float b = (float(x)+0.5f)/8.f, c = (float(y)+0.5f)/8.f;
          const float denominator = (1.f-b-c)/w[0] + b/w[1] + c/w[2];
          const float value = linear ? (1.f-b-c)*0.25f + b*0.5f + c*0.75f
            : ((1.f-b-c)*0.25f/w[0] + b*0.5f/w[1] + c*0.75f/w[2])/denominator;
          reference[y*4+x] = {uint8_t(value*255.f+0.5f),64,0,255};
        }
      }
      const auto data = record_for(w, values);
      gpu.Render({}, layout, BuildMeshEmulationVertexShader(layout, 0), old.spirv,
        data, {}, &tess, {}, reference);
      gpu.Render({}, layout, BuildMeshEmulationVertexShader(layout, 0, params), updated.spirv,
        data, {}, nullptr, {}, reference);
      ++cases;
    }
  }
  // Exercise the separate native VS producer with duplicate indexed vertices and a nonzero
  // first instance. Both instances must have their own records, even when vertex caching differs.
  const uint32_t capture_code[]{
    EncodeVop1(0x01, 2, 255), 0xbf800000u,
    EncodeVop1(0x01, 3, 255), 0xbf800000u,
    EncodeVop1(0x01, 4, 255), 0x40400000u,
    EncodeVopc(0xc2, 129, 5), EncodeVop2(0x01, 2, 258, 4),
    EncodeVopc(0xc2, 130, 5), EncodeVop2(0x01, 3, 259, 4),
    EncodeVop1(0x01, 0, 128), EncodeVop1(0x01, 1, 242),
    EncodeExp0(0x0c, 0xf), EncodeExp1(2,3,0,1),
    EncodeVop1(0x01, 10, 255), packed[0],
    EncodeVop1(0x01, 20, 255), packed[1],
    EncodeVopc(0xc2, 129, 5), EncodeVop2(0x01, 10, 266, 20),
    EncodeVop1(0x01, 20, 255), packed[2],
    EncodeVopc(0xc2, 130, 5), EncodeVop2(0x01, 10, 266, 20),
    EncodeExp0(0x20, 0xf), EncodeExp1(10,10,10,10), EncodeSopp(0x01)};
  auto capture_options = MakeCompileOptions(ShaderType::Vertex);
  auto capture_profile = Spirv::DefaultProfile;
  capture_profile.vertex_subgroups = false;
  capture_options.profile = &capture_profile;
  const auto capture = RecompileForVertexStoreTest(capture_code, capture_options,
    Spirv::PvStoreInfo{.vertex_store = true});
  const auto capture_layout = BuildVertexStoreLayout(capture.program.info.outputs);
  const std::array<uint32_t, 6> capture_indices{0,1,2,0,1,2};
  const auto plan = PlanVertexStore(capture_indices, 0, 2, capture_layout.vertex_dwords, 4096);
  Check(plan.valid, "native capture fixture does not fit");
  std::vector<uint32_t> initial(plan.bytes / 4u + 16u, 0xcdcdcdcdu);
  initial[VertexStoreFirst] = plan.first;
  initial[VertexStoreSpan] = plan.span;
  initial[VertexStoreFirstInstance] = 4u;
  initial[VertexStoreIndexCount] = uint32_t(capture_indices.size());
  initial[VertexStoreStride] = capture_layout.vertex_dwords;
  initial[VertexStoreRecords] = plan.records_offset;
  std::ranges::copy(plan.indices, initial.begin() + VertexStoreHeaderDwords);
  pixel.input_num = 1;
  pixel.custom_interpolation_mask = 1;
  pixel.interpolator_settings[0] = 0;
  pixel.parameter_mode = ShaderPixelParameterMode::FirstVertex;
  profile.vertex_store = true;
  options.input_info.pixel = &pixel;
  const auto capture_fragment = RecompileForVertexStoreTest(code, options, profile);
  const auto capture_params = TriangleParameters(pixel, capture_fragment.program.info);
  const auto consumer = BuildVertexStoreVertexShader(capture_layout, 0, capture_params);
  gpu.Render({}, capture_layout, consumer, capture_fragment.spirv, initial, {0,255,0,255},
    nullptr, {}, {}, &capture.spirv, uint32_t(capture_indices.size()), 2u, 4u);
  ++cases;
  std::printf("VS-store GPU: %u cases passed (native capture, packed words, cull, first/last, perspective/linear W)\n", cases);
}
