// Six triangles have distinct group/instance words and unequal clip W. The PS reads
// all three packed corners and perspective inputs in the same draw.
void TestMeshBatchCornersGpu() {
  using namespace ShaderRecompiler;
  MeshGpuTest gpu;
  const std::array<uint32_t, 3> words{0x7fc01234u, 0xff801234u, 0x12345678u};
  const uint32_t producer_code[]{
    EncodeSMovB32(124, 255), 0x1003u, EncodeSopp(0x10, 9),
    EncodeVop1(0x01, 30, 261), EncodeVop1(0x01, 31, 262),
    EncodeVop1(0x06, 10, 261), EncodeVop1(0x06, 11, 262),
    EncodeVop2(0x08, 10, 255, 10), 0x3f000000u,
    EncodeVop2(0x08, 11, 255, 11), 0x3f000000u,
    EncodeVop2(0x03, 10, 255, 10), 0xbf4ccccdu,
    EncodeVop2(0x03, 11, 255, 11), 0xbf4ccccdu,
    0xd7650001u, 0x000100c1u,
    EncodeVop1(0x01, 2, 255), 0xbf800000u,
    EncodeVop1(0x01, 3, 255), 0xbf800000u,
    EncodeVop1(0x01, 4, 255), 0x40400000u,
    EncodeVopc(0xc2, 129, 1), EncodeVop2(0x01, 2, 258, 4),
    EncodeVopc(0xc2, 130, 1), EncodeVop2(0x01, 3, 259, 4),
    EncodeVop2(0x08, 2, 255, 2), 0x3e4ccccdu,
    EncodeVop2(0x08, 3, 255, 3), 0x3e4ccccdu,
    EncodeVop2(0x03, 2, 266, 2), EncodeVop2(0x03, 3, 267, 3),
    EncodeVop1(0x01, 20, 255), words[0],
    EncodeVop1(0x01, 21, 255), words[1],
    EncodeVop1(0x01, 22, 255), words[2],
    EncodeVop1(0x01, 24, 242),
    EncodeVop1(0x01, 25, 255), 0x40000000u,
    EncodeVop1(0x01, 26, 255), 0x40800000u,
    EncodeVopc(0xc2, 129, 1),
    EncodeVop2(0x01, 20, 276, 21), EncodeVop2(0x01, 24, 280, 25),
    EncodeVopc(0xc2, 130, 1),
    EncodeVop2(0x01, 20, 276, 22), EncodeVop2(0x01, 24, 280, 26),
    EncodeVop2(0x08, 2, 258, 24), EncodeVop2(0x08, 3, 259, 24),
    EncodeVop1(0x01, 23, 128),
    EncodeExp0(0x0c, 0xf, false), EncodeExp1(2, 3, 23, 24),
    EncodeExp0(0x20, 0xf, false), EncodeExp1(30, 31, 20, 20),
    EncodeVop1(0x01, 9, 255), 0x00200400u,
    EncodeExp0(0x14, 0x1), EncodeExp1(9, 0, 0, 0), EncodeSopp(0x01)};
  ShaderVertexInputInfo input{};
  input.logical_stage = ShaderType::Mesh;
  auto& mesh = input.mesh;
  mesh.emulated = mesh.fast_launch = true;
  mesh.input_primitive = static_cast<uint32_t>(Prospero::PrimitiveType::kPointList);
  mesh.wave_size = 32;
  mesh.host_subgroup_size = gpu.subgroup_size;
  mesh.threads_num[0] = 32;
  mesh.threads_num[1] = mesh.threads_num[2] = 1;
  mesh.max_vertices = mesh.vertices_per_group = 3;
  mesh.max_primitives = mesh.primitives_per_group = 1;
  CompileOptions producer_options{};
  producer_options.stage = ShaderType::Mesh;
  producer_options.wave_size = 32;
  producer_options.input_info.vertex = &input;
  const auto prefix_dwords = IR::PushData::GraphicsDrawDwords(true, false, 0);
  const auto header_dword = MeshEmulationHeaderDword(0);
  const auto producer = RecompileForTest(producer_code, producer_options,
    prefix_dwords + (header_dword != IR::PushData::NoStart ? MeshEmulationLayout::HeaderDwords : 0u));
  const auto layout = BuildMeshEmulationLayout(producer.program.info.outputs, input);

  std::vector<uint32_t> pixel_code{EncodeVop1(0x01, 30, 256), EncodeVop1(0x01, 10, 242)};
  for (uint32_t corner = 0; corner < 3; ++corner) {
    const uint32_t check[]{EncodeVintrp(2, 12, 0, 2, (corner + 2u) % 3u),
      EncodeVop1(0x01, 20, 255), words[corner],
      EncodeVopc(0xc2, 268, 20), EncodeVop2(0x01, 10, 128, 10)};
    pixel_code.insert(pixel_code.end(), std::begin(check), std::end(check));
  }
  const uint32_t finish[]{
    EncodeVintrp(2, 12, 0, 0, 2), EncodeVintrp(2, 13, 0, 1, 2),
    EncodeVop1(0x06, 12, 268), EncodeVop1(0x06, 13, 269),
    EncodeVop2(0x08, 12, 255, 12), 0x3e800000u,
    EncodeVop2(0x08, 13, 255, 13), 0x3e800000u,
    EncodeExp0(0x00, 0xf), EncodeExp1(12, 13, 10, 30), EncodeSopp(0x01)};
  pixel_code.insert(pixel_code.end(), std::begin(finish), std::end(finish));
  ShaderPixelInputInfo pixel{};
  pixel.input_num = 1;
  pixel.custom_interpolation_mask = 1;
  pixel.emulate_per_vertex = true;
  pixel.ps_perspective_center_vgpr = 0;
  pixel.ps_system_input_base = 2;
  pixel.target_output_mode[0] = 4;
  auto pixel_options = MakeCompileOptions(ShaderType::Pixel);
  pixel_options.input_info.pixel = &pixel;
  const auto old_fragment = RecompileForTest(pixel_code, pixel_options);
  const Spirv::PvStoreInfo profile{.per_vertex_buffer = true};
  const auto buffer_fragment = RecompileForVertexStoreTest(pixel_code, pixel_options, profile);
  const auto parameters = TriangleParameters(pixel, old_fragment.program.info);
  const auto tessellation = BuildTriangleShaders(parameters, 0, 0);
  const auto legacy_vertex = BuildMeshEmulationVertexShader(layout, 0);
  const auto buffer_vertex = BuildMeshEmulationVertexShader(layout, 0, parameters);
  MeshGpuBatchCase legacy{2, 3, 6, 65535, 65535, 3, true};
  auto batched = legacy;
  batched.legacy = false;
  batched.expected_batches = MeshEmulationBatchEnabled() ? 1u : 3u;
  std::vector<uint32_t> initial(4u + 6u * layout.GroupDwords() + 16u, 0xcdcdcdcdu);
  initial[0] = initial[1] = 0;
  initial[2] = 2;
  initial[3] = 3;
  std::vector<uint32_t> prefix(prefix_dwords, 0u);
  prefix[0] = 2;
  std::array<std::array<uint8_t, 4>, 16> expected;
  expected.fill({0, 0, 0, 255});
  for (uint32_t instance = 0; instance < 3; ++instance)
    for (uint32_t group = 0; group < 2; ++group)
      expected[instance * 4u + group] = {uint8_t(group * 64u), uint8_t((instance * 255u + 2u) / 4u), 255, 65};
  const auto reference = gpu.Render(producer.spirv, layout, legacy_vertex, old_fragment.spirv,
    initial, {}, &tessellation, prefix, expected, &legacy);
  const auto* switch_value = std::getenv("KYTY_PERF_VS_STORE");
  const bool store = switch_value == nullptr || std::strcmp(switch_value, "0") != 0;
  const auto actual = gpu.Render(producer.spirv, layout, store ? buffer_vertex : legacy_vertex,
    store ? buffer_fragment.spirv : old_fragment.spirv, initial, {}, store ? nullptr : &tessellation,
    prefix, expected, &batched);
  Check(actual == reference, "2x3 batched raw corners or perspective pixels differ from legacy tessellation");
  std::printf("Mesh batch corners GPU: 2 groups x 3 instances, batch=%u store=%u, exact legacy pixels\n",
    unsigned(MeshEmulationBatchEnabled()), unsigned(store));
}
