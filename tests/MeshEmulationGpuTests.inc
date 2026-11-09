// A small portability harness: no mesh shaders, native null descriptors or int64 atomics.
class MeshGpuTest {
public:
  MeshGpuTest() {
    const auto* library = std::getenv("KYTY_TEST_VULKAN_LIBRARY");
    loader = std::make_unique<vk::detail::DynamicLoader>(library ? library : "");
    const auto proc = loader->getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    Check(proc != nullptr, "portable GPU test could not load Vulkan");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(proc);
    vk::ApplicationInfo app{.pApplicationName = "MeshEmulationGpuTests", .apiVersion = VK_API_VERSION_1_3};
    vk::InstanceCreateInfo create{.pApplicationInfo = &app};
#if defined(__APPLE__)
    const char* instance_extension = VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
    const auto extensions = EnumerateVulkan<vk::ExtensionProperties>("test instance extensions",
      [&](uint32_t* n, vk::ExtensionProperties* p) { return vk::enumerateInstanceExtensionProperties(nullptr, n, p); });
    if (std::ranges::any_of(extensions, [&](const auto& extension) {
          return std::strcmp(extension.extensionName.data(), instance_extension) == 0;
        })) {
      create.flags = vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
      create.enabledExtensionCount = 1;
      create.ppEnabledExtensionNames = &instance_extension;
    }
#endif
    Ok(vk::createInstance(&create, nullptr, &instance));
    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance);
    const auto devices = EnumerateVulkan<vk::PhysicalDevice>("test devices", [&](uint32_t* n, vk::PhysicalDevice* p) {
      return instance.enumeratePhysicalDevices(n, p);
    });
    Check(!devices.empty(), "portable GPU test found no device");
    physical = devices.front();
    const auto queues = physical.getQueueFamilyProperties();
    while (family < queues.size() && (queues[family].queueFlags &
           (vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute)) !=
           (vk::QueueFlagBits::eGraphics | vk::QueueFlagBits::eCompute)) ++family;
    Check(family < queues.size(), "portable GPU test found no graphics+compute queue");
    vk::PhysicalDeviceSubgroupProperties subgroup{};
    vk::PhysicalDeviceProperties2 properties{.pNext = &subgroup};
    physical.getProperties2(&properties);
    subgroup_size = subgroup.subgroupSize;
    std::printf("Portable GPU: %s, subgroup %u\n", properties.properties.deviceName.data(), subgroup_size);
    vk::PhysicalDeviceFeatures core{};
    core.tessellationShader = true;
    core.shaderInt64 = true;
    vk::PhysicalDeviceVulkan12Features features12{.scalarBlockLayout = true};
    vk::PhysicalDeviceVulkan13Features features13{.pNext = &features12,
      .synchronization2 = true, .dynamicRendering = true};
    const float priority = 1.f;
    vk::DeviceQueueCreateInfo queue_info{.queueFamilyIndex = family, .queueCount = 1, .pQueuePriorities = &priority};
    vk::DeviceCreateInfo device_info{.pNext = &features13, .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info, .pEnabledFeatures = &core};
#if defined(__APPLE__)
    const char* device_extension = "VK_KHR_portability_subset";
    device_info.enabledExtensionCount = 1;
    device_info.ppEnabledExtensionNames = &device_extension;
#endif
    Ok(physical.createDevice(&device_info, nullptr, &device));
    VULKAN_HPP_DEFAULT_DISPATCHER.init(device);
    device.getQueue(family, 0, &queue);
    vk::CommandPoolCreateInfo pool_info{.queueFamilyIndex = family};
    Ok(device.createCommandPool(&pool_info, nullptr, &pool));
  }

  ~MeshGpuTest() {
    device.destroyCommandPool(pool);
    device.destroy();
    instance.destroy();
  }

  void Render(const std::vector<uint32_t>& compute, const MeshEmulationLayout& layout,
              const std::vector<uint32_t>& vertex, const std::vector<uint32_t>& fragment,
              std::span<const uint32_t> initial, std::array<uint8_t, 4> expected,
              const RectListShaders* tessellation = nullptr,
              std::span<const uint32_t> push_data = {},
              std::span<const std::array<uint8_t, 4>> expected_pixels = {}) {
    constexpr uint32_t width = 4, height = 4;
    const auto memory = physical.getMemoryProperties();
    const auto allocate = [&](vk::MemoryRequirements requirements, vk::MemoryPropertyFlags flags) {
      uint32_t type = 0;
      while (type < memory.memoryTypeCount &&
             (!(requirements.memoryTypeBits & (1u << type)) ||
              (memory.memoryTypes[type].propertyFlags & flags) != flags)) ++type;
      Check(type < memory.memoryTypeCount, "portable GPU memory type unavailable");
      vk::MemoryAllocateInfo info{.allocationSize = requirements.size, .memoryTypeIndex = type};
      vk::DeviceMemory result;
      Ok(device.allocateMemory(&info, nullptr, &result));
      return result;
    };
    struct BufferAllocation { vk::Buffer buffer; vk::DeviceMemory memory; void* mapped; };
    const auto buffer = [&](uint64_t size) {
      BufferAllocation result{};
      vk::BufferCreateInfo info{.size = size, .usage = vk::BufferUsageFlagBits::eStorageBuffer |
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst};
      Ok(device.createBuffer(&info, nullptr, &result.buffer));
      result.memory = allocate(device.getBufferMemoryRequirements(result.buffer),
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
      Ok(device.bindBufferMemory(result.buffer, result.memory, 0));
      Ok(device.mapMemory(result.memory, 0, size, {}, &result.mapped));
      return result;
    };
    auto output = buffer(initial.size_bytes());
    std::memcpy(output.mapped, initial.data(), initial.size_bytes());
    auto pixels = buffer(width * height * 4u);
    std::memset(pixels.mapped, 0xcd, width * height * 4u);
    vk::ImageCreateInfo image_info{.imageType = vk::ImageType::e2D,
      .format = vk::Format::eR8G8B8A8Unorm, .extent = {width, height, 1}, .mipLevels = 1,
      .arrayLayers = 1, .samples = vk::SampleCountFlagBits::e1,
      .tiling = vk::ImageTiling::eOptimal,
      .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc};
    vk::Image image;
    Ok(device.createImage(&image_info, nullptr, &image));
    auto image_memory = allocate(device.getImageMemoryRequirements(image), {});
    Ok(device.bindImageMemory(image, image_memory, 0));
    vk::ImageViewCreateInfo view_info{.image = image, .viewType = vk::ImageViewType::e2D,
      .format = image_info.format, .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};
    vk::ImageView view;
    Ok(device.createImageView(&view_info, nullptr, &view));
    const vk::DescriptorSetLayoutBinding binding{MeshEmulationBinding, vk::DescriptorType::eStorageBuffer,
      1, vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eVertex, nullptr};
    vk::DescriptorSetLayoutCreateInfo set_info{.bindingCount = 1, .pBindings = &binding};
    vk::DescriptorSetLayout set_layout;
    Ok(device.createDescriptorSetLayout(&set_info, nullptr, &set_layout));
    const vk::PushConstantRange push{vk::ShaderStageFlagBits::eCompute, 0, 128};
    vk::PipelineLayoutCreateInfo pipeline_layout_info{.setLayoutCount = 1,
      .pSetLayouts = &set_layout, .pushConstantRangeCount = 1, .pPushConstantRanges = &push};
    vk::PipelineLayout pipeline_layout;
    Ok(device.createPipelineLayout(&pipeline_layout_info, nullptr, &pipeline_layout));
    const vk::DescriptorPoolSize pool_size{vk::DescriptorType::eStorageBuffer, 1};
    vk::DescriptorPoolCreateInfo descriptor_pool_info{.maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &pool_size};
    vk::DescriptorPool descriptor_pool;
    Ok(device.createDescriptorPool(&descriptor_pool_info, nullptr, &descriptor_pool));
    vk::DescriptorSetAllocateInfo set_allocate{.descriptorPool = descriptor_pool,
      .descriptorSetCount = 1, .pSetLayouts = &set_layout};
    vk::DescriptorSet set;
    Ok(device.allocateDescriptorSets(&set_allocate, &set));
    const vk::DescriptorBufferInfo buffer_info{output.buffer, 0, initial.size_bytes()};
    vk::WriteDescriptorSet write{.dstSet = set, .dstBinding = MeshEmulationBinding,
      .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &buffer_info};
    device.updateDescriptorSets(1, &write, 0, nullptr);
    std::vector<vk::ShaderModule> modules;
    const auto module = [&](const std::vector<uint32_t>& words) {
      CheckSpirvBinaryValidates(words);
      modules.push_back(CompileSPV(words, device));
      return modules.back();
    };
    vk::Pipeline compute_pipeline{};
    if (!compute.empty()) {
      vk::ComputePipelineCreateInfo info{.stage = {.stage = vk::ShaderStageFlagBits::eCompute,
        .module = module(compute), .pName = "main"}, .layout = pipeline_layout};
      Ok(device.createComputePipelines({}, 1, &info, nullptr, &compute_pipeline));
    }
    std::vector<vk::PipelineShaderStageCreateInfo> stages{
      {.stage = vk::ShaderStageFlagBits::eVertex, .module = module(vertex), .pName = "main"},
      {.stage = vk::ShaderStageFlagBits::eFragment, .module = module(fragment), .pName = "main"}};
    if (tessellation) {
      stages.push_back({.stage = vk::ShaderStageFlagBits::eTessellationControl,
        .module = module(tessellation->control), .pName = "main"});
      stages.push_back({.stage = vk::ShaderStageFlagBits::eTessellationEvaluation,
        .module = module(tessellation->evaluation), .pName = "main"});
    }
    const vk::PipelineVertexInputStateCreateInfo vertex_input{};
    const vk::PipelineInputAssemblyStateCreateInfo assembly{.topology = tessellation
      ? vk::PrimitiveTopology::ePatchList : vk::PrimitiveTopology::eTriangleList};
    const vk::PipelineTessellationStateCreateInfo tessellation_info{.patchControlPoints = 3};
    const vk::Viewport viewport{0, 0, float(width), float(height), 0, 1};
    const vk::Rect2D scissor{{0, 0}, {width, height}};
    const vk::PipelineViewportStateCreateInfo viewport_info{.viewportCount = 1, .pViewports = &viewport,
      .scissorCount = 1, .pScissors = &scissor};
    const vk::PipelineRasterizationStateCreateInfo raster{.polygonMode = vk::PolygonMode::eFill,
      .cullMode = vk::CullModeFlagBits::eNone, .frontFace = vk::FrontFace::eCounterClockwise, .lineWidth = 1};
    const vk::PipelineMultisampleStateCreateInfo samples{.rasterizationSamples = vk::SampleCountFlagBits::e1};
    const vk::PipelineColorBlendAttachmentState attachment{.colorWriteMask =
      vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
      vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};
    const vk::PipelineColorBlendStateCreateInfo blend{.attachmentCount = 1, .pAttachments = &attachment};
    const vk::PipelineRenderingCreateInfo rendering{.colorAttachmentCount = 1, .pColorAttachmentFormats = &image_info.format};
    const vk::GraphicsPipelineCreateInfo graphics_info{.pNext = &rendering,
      .stageCount = uint32_t(stages.size()), .pStages = stages.data(), .pVertexInputState = &vertex_input,
      .pInputAssemblyState = &assembly, .pTessellationState = tessellation ? &tessellation_info : nullptr,
      .pViewportState = &viewport_info, .pRasterizationState = &raster, .pMultisampleState = &samples,
      .pColorBlendState = &blend, .layout = pipeline_layout};
    vk::Pipeline graphics_pipeline;
    Ok(device.createGraphicsPipelines({}, 1, &graphics_info, nullptr, &graphics_pipeline));
    vk::CommandBufferAllocateInfo command_info{.commandPool = pool,
      .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1};
    vk::CommandBuffer command;
    Ok(device.allocateCommandBuffers(&command_info, &command));
    vk::CommandBufferBeginInfo begin{};
    Ok(command.begin(&begin));
    if (compute_pipeline) {
      command.bindPipeline(vk::PipelineBindPoint::eCompute, compute_pipeline);
      command.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipeline_layout, 0, 1, &set, 0, nullptr);
      if (!push_data.empty()) command.pushConstants(pipeline_layout, vk::ShaderStageFlagBits::eCompute,
        0, uint32_t(push_data.size_bytes()), push_data.data());
      command.dispatch(1, 1, 1);
      vk::MemoryBarrier barrier{.srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eShaderRead};
      command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eVertexShader, {}, 1, &barrier, 0, nullptr, 0, nullptr);
    }
    vk::ImageMemoryBarrier to_color{.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite,
      .oldLayout = vk::ImageLayout::eUndefined, .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image, .subresourceRange = view_info.subresourceRange};
    command.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
      vk::PipelineStageFlagBits::eColorAttachmentOutput, {}, 0, nullptr, 0, nullptr, 1, &to_color);
    vk::RenderingAttachmentInfo color{.imageView = view, .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
      .loadOp = vk::AttachmentLoadOp::eClear, .storeOp = vk::AttachmentStoreOp::eStore,
      .clearValue = {.color = {.float32 = std::array<float, 4>{0.f, 0.f, 0.f, 1.f}}}};
    vk::RenderingInfo render{.renderArea = scissor, .layerCount = 1, .colorAttachmentCount = 1,
      .pColorAttachments = &color};
    command.beginRendering(&render);
    command.bindPipeline(vk::PipelineBindPoint::eGraphics, graphics_pipeline);
    command.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline_layout, 0, 1, &set, 0, nullptr);
    command.draw(layout.max_primitives * 3u, 1, 0, 0);
    command.endRendering();
    vk::ImageMemoryBarrier to_copy = to_color;
    to_copy.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
    to_copy.dstAccessMask = vk::AccessFlagBits::eTransferRead;
    to_copy.oldLayout = vk::ImageLayout::eColorAttachmentOptimal;
    to_copy.newLayout = vk::ImageLayout::eTransferSrcOptimal;
    command.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
      vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 0, nullptr, 1, &to_copy);
    const vk::BufferImageCopy copy{.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
      .imageExtent = {width, height, 1}};
    command.copyImageToBuffer(image, vk::ImageLayout::eTransferSrcOptimal, pixels.buffer, 1, &copy);
    vk::MemoryBarrier host_barrier{.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite,
      .dstAccessMask = vk::AccessFlagBits::eHostRead};
    command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
      vk::PipelineStageFlagBits::eHost, {}, 1, &host_barrier, 0, nullptr, 0, nullptr);
    Ok(command.end());
    vk::SubmitInfo submit{.commandBufferCount = 1, .pCommandBuffers = &command};
    Ok(queue.submit(1, &submit, {}));
    Ok(queue.waitIdle());
    const auto* record = static_cast<const uint32_t*>(output.mapped);
    if (compute_pipeline && expected[0] == 255u) {
      Check(record[4] == 3u && record[5] == 1u, "mesh allocation readback differs");
      const std::array<float, 12> positions{-1,-1,0,1, 3,-1,0,1, -1,3,0,1};
      const auto field = std::ranges::find_if(layout.fields, [](const auto& f) {
        return f.output.kind == ShaderRecompiler::IR::StageOutputKind::Position;
      });
      Check(field != layout.fields.end(), "GPU fixture lost position layout");
      for (uint32_t v = 0; v < 3; ++v)
        for (uint32_t c = 0; c < 4; ++c) {
          const auto actual = record[4 + layout.VerticesOffset() + v * layout.vertex_dwords + field->offset + c];
          if (actual != std::bit_cast<uint32_t>(positions[v*4+c]))
            std::fprintf(stderr, "Position v%u.%u: got %08x expected %08x\n", v, c,
              actual, std::bit_cast<uint32_t>(positions[v*4+c]));
          Check(actual == std::bit_cast<uint32_t>(positions[v*4+c]), "mesh position readback differs");
        }
    }
    const auto* read_pixels = static_cast<const uint8_t*>(pixels.mapped);
    Check(expected_pixels.empty() || expected_pixels.size() == width * height,
          "portable GPU reference image has the wrong size");
    for (uint32_t p = 0; p < width * height; ++p) {
      if (!expected_pixels.empty()) expected = expected_pixels[p];
      const bool matches = std::equal(expected.begin(), expected.end(), read_pixels + p * 4u,
        [&](uint8_t reference, uint8_t actual) {
          return std::abs(int(reference) - int(actual)) <= (expected_pixels.empty() ? 0 : 1);
        });
      if (!matches) {
        std::fprintf(stderr, "Pixel %u: got %u,%u,%u,%u expected %u,%u,%u,%u\n", p,
          read_pixels[p*4], read_pixels[p*4+1], read_pixels[p*4+2], read_pixels[p*4+3],
          expected[0], expected[1], expected[2], expected[3]);
        const auto* diagnostics = static_cast<const uint32_t*>(output.mapped);
        std::fprintf(stderr, "Mesh record counts=%u,%u primitive=%08x\n",
          diagnostics[4], diagnostics[5], diagnostics[6]);
        Check(false, "portable GPU pixel mismatch");
      }
    }
    for (uint32_t i = MeshEmulationLayout::HeaderDwords + layout.GroupDwords(); i < initial.size(); ++i)
      Check(record[i] == initial[i], "mesh compute wrote beyond its output record");
    device.freeCommandBuffers(pool, 1, &command);
    device.destroyPipeline(graphics_pipeline);
    device.destroyPipeline(compute_pipeline);
    for (auto shader : modules) device.destroyShaderModule(shader);
    device.destroyDescriptorPool(descriptor_pool);
    device.destroyPipelineLayout(pipeline_layout);
    device.destroyDescriptorSetLayout(set_layout);
    device.destroyImageView(view);
    device.destroyImage(image);
    device.freeMemory(image_memory);
    for (auto allocation : {output, pixels}) {
      device.unmapMemory(allocation.memory);
      device.destroyBuffer(allocation.buffer);
      device.freeMemory(allocation.memory);
    }
  }

  uint32_t subgroup_size = 0;
private:
  static void Ok(vk::Result result) {
    if (result != vk::Result::eSuccess) std::fprintf(stderr, "Vulkan: %s\n", vk::to_string(result).c_str());
    Check(result == vk::Result::eSuccess, "portable Vulkan operation failed");
  }
  std::unique_ptr<vk::detail::DynamicLoader> loader;
  vk::Instance instance;
  vk::PhysicalDevice physical;
  vk::Device device;
  vk::Queue queue;
  vk::CommandPool pool;
  uint32_t family = 0;
};

void TestMeshEmulationGpu() {
  MeshGpuTest gpu;
  const uint32_t pixel_code[]{
    EncodeVop1(0x01, 0, 242), EncodeVop1(0x01, 1, 128),
    EncodeExp0(0x00, 0xf), EncodeExp1(0, 1, 1, 0), EncodeSopp(0x01)};
  ShaderPixelInputInfo pixel{};
  pixel.target_output_mode[0] = 4;
  auto pixel_options = MakeCompileOptions(ShaderType::Pixel);
  pixel_options.input_info.pixel = &pixel;
  const auto fragment = RecompileForTest(pixel_code, pixel_options).spirv;
  uint32_t cases = 0;
  for (const auto wave : {32u, 64u}) {
    for (const auto packed : {0x00200400u, 0x00200600u, 0x80200400u, 0x00400400u}) {
      const uint32_t code[]{
        EncodeSMovB32(124, 255), 0x1003u, EncodeSopp(0x10, 9),
        0xd7650001u, 0x000100c1u, // logical lane index
        EncodeVop1(0x01, 2, 255), 0xbf800000u,
        EncodeVop1(0x01, 3, 255), 0xbf800000u,
        EncodeVop1(0x01, 4, 255), 0x40400000u,
        EncodeVopc(0xc2, 129, 1), EncodeVop2(0x01, 2, 258, 4),
        EncodeVopc(0xc2, 130, 1), EncodeVop2(0x01, 3, 259, 4),
        EncodeVop1(0x01, 5, 128), EncodeVop1(0x01, 6, 242),
        EncodeExp0(0x0c, 0xf, false), EncodeExp1(2, 3, 5, 6),
        EncodeVop1(0x01, 9, 255), packed,
        EncodeExp0(0x14, 0x1), EncodeExp1(9, 0, 0, 0), EncodeSopp(0x01)};
      ShaderVertexInputInfo input{};
      input.logical_stage = ShaderType::Mesh;
      auto& mesh = input.mesh;
      mesh.emulated = true;
      mesh.wave_size = wave;
      mesh.host_subgroup_size = gpu.subgroup_size;
      mesh.threads_num[0] = wave;
      mesh.threads_num[1] = mesh.threads_num[2] = 1;
      mesh.max_vertices = mesh.vertices_per_group = 3;
      mesh.max_primitives = mesh.primitives_per_group = 1;
      ShaderRecompiler::CompileOptions options{};
      options.stage = ShaderType::Mesh;
      options.wave_size = wave;
      options.input_info.vertex = &input;
      const auto result = RecompileForTest(code, options);
      const auto layout = BuildMeshEmulationLayout(result.program.info.outputs, input);
      const auto vertex = BuildMeshEmulationVertexShader(layout, 0);
      std::vector<uint32_t> initial(4u + layout.GroupDwords() + 16u, 0xcdcdcdcdu);
      initial[0] = initial[1] = 0;
      initial[2] = initial[3] = 1;
      const bool visible = packed == 0x00200400u || packed == 0x00200600u;
      const std::array<uint32_t, 5> prefix{3, 0, 0, 0, 0};
      gpu.Render(result.spirv, layout, vertex, fragment, initial,
        {uint8_t(visible ? 255 : 0), 0, 0, 255}, nullptr, prefix);
      ++cases;
    }
  }
  std::printf("Mesh GPU: %u cases passed, exact pixels and output records\n", cases);

  // Packed corner values include a NaN payload. Any float interpolation corrupts their bits.
  const std::array<uint32_t, 3> words{0x7fc01234u, 0xff801234u, 0x12345678u};
  std::vector<uint32_t> corner_code{EncodeVop1(0x01, 10, 242)};
  for (uint32_t corner = 0; corner < 3; ++corner) {
    const std::array<uint32_t, 6> instructions{
      EncodeVintrp(2, 12, 0, 0, (corner + 2u) % 3u),
      EncodeVop1(0x01, 20, 255), words[corner],
      EncodeVopc(0xc2, 268, 20), EncodeVop2(0x01, 10, 128, 10),
      EncodeSopp(0x00)};
    corner_code.insert(corner_code.end(), instructions.begin(), instructions.end());
  }
  const std::array<uint32_t, 5> finish{EncodeVop1(0x01, 0, 128), EncodeVop1(0x01, 1, 242),
    EncodeExp0(0x00, 0xf), EncodeExp1(0, 10, 0, 1), EncodeSopp(0x01)};
  corner_code.insert(corner_code.end(), finish.begin(), finish.end());
  ShaderPixelInputInfo corner_pixel{};
  corner_pixel.input_num = 1;
  corner_pixel.custom_interpolation_mask = 1;
  corner_pixel.emulate_per_vertex = true;
  corner_pixel.target_output_mode[0] = 4;
  auto corner_options = MakeCompileOptions(ShaderType::Pixel);
  corner_options.input_info.pixel = &corner_pixel;
  const auto corner_fragment = RecompileForTest(corner_code, corner_options);
  const auto parameters = TriangleParameters(corner_pixel, corner_fragment.program.info);
  const auto tessellation = BuildTriangleShaders(parameters, 0, 0);
  ShaderVertexInputInfo corner_input{};
  corner_input.mesh.max_vertices = 3;
  corner_input.mesh.max_primitives = 1;
  const std::vector<ShaderRecompiler::IR::StageOutput> fields{
    {ShaderRecompiler::IR::StageOutputKind::Position, 0, 0, "position"},
    {ShaderRecompiler::IR::StageOutputKind::Parameter, 0, 0, "packed"}};
  const auto corner_layout = BuildMeshEmulationLayout(fields, corner_input);
  std::vector<uint32_t> record(4 + corner_layout.GroupDwords() + 16, 0xcdcdcdcdu);
  record[0] = record[1] = 0;
  record[2] = record[3] = 1;
  record[4] = 3;
  record[5] = 1;
  record[6] = 0x00200400u;
  const std::array<float, 12> positions{-1,-1,0,1, 3,-1,0,1, -1,3,0,1};
  for (uint32_t v = 0; v < 3; ++v) {
    const auto base = 4 + corner_layout.VerticesOffset() + v * corner_layout.vertex_dwords;
    for (uint32_t c = 0; c < 4; ++c) {
      record[base + c] = std::bit_cast<uint32_t>(positions[v*4+c]);
      record[base + 4 + c] = words[v];
    }
  }
  gpu.Render({}, corner_layout, BuildMeshEmulationVertexShader(corner_layout, 0),
    corner_fragment.spirv, record, {0, 255, 0, 255}, &tessellation);
  std::puts("Triangle GPU: packed corner words survived tessellation and fragment reads");

  // Check the interpolated basis at every pixel, including perspective correction.
  const uint32_t bary_code[]{EncodeVop1(0x01, 2, 128), EncodeVop1(0x01, 3, 242),
    EncodeExp0(0x00, 0xf), EncodeExp1(0, 1, 2, 3), EncodeSopp(0x01)};
  ShaderPixelInputInfo bary_pixel{};
  bary_pixel.emulate_per_vertex = true;
  bary_pixel.ps_perspective_center_vgpr = 0;
  bary_pixel.ps_system_input_base = 2;
  bary_pixel.target_output_mode[0] = 4;
  auto bary_options = MakeCompileOptions(ShaderType::Pixel);
  bary_options.input_info.pixel = &bary_pixel;
  const auto bary_fragment = RecompileForTest(bary_code, bary_options);
  Check(!SpirvContainsCapability(bary_fragment.spirv, 5284u),
        "portable GPU fixture still requires native fragment barycentrics");
  const auto bary_tessellation = BuildTriangleShaders(
    TriangleParameters(bary_pixel, bary_fragment.program.info), 0, 0);
  for (const bool perspective : {false, true}) {
    auto bary_record = record;
    const std::array<float, 3> w{1.f, perspective ? 2.f : 1.f, perspective ? 4.f : 1.f};
    for (uint32_t v = 0; v < 3; ++v) {
      const auto base = 4 + corner_layout.VerticesOffset() + v * corner_layout.vertex_dwords;
      for (uint32_t c = 0; c < 4; ++c)
        bary_record[base + c] = std::bit_cast<uint32_t>(positions[v*4+c] * w[v]);
    }
    std::array<std::array<uint8_t, 4>, 16> reference{};
    for (uint32_t y = 0; y < 4; ++y) {
      for (uint32_t x = 0; x < 4; ++x) {
        const float b = (float(x) + 0.5f) / 8.f, c = (float(y) + 0.5f) / 8.f;
        const float denominator = (1.f - b - c) / w[0] + b / w[1] + c / w[2];
        reference[y*4+x] = {uint8_t(b / w[1] / denominator * 255.f + 0.5f),
                            uint8_t(c / w[2] / denominator * 255.f + 0.5f), 0, 255};
      }
    }
    gpu.Render({}, corner_layout, BuildMeshEmulationVertexShader(corner_layout, 0),
      bary_fragment.spirv, bary_record, {}, &bary_tessellation, {}, reference);
  }
  std::puts("Triangle GPU: barycentric coordinates match all pixels with equal and unequal clip W");

  auto cull_fields = fields;
  cull_fields.push_back({ShaderRecompiler::IR::StageOutputKind::CullDistance, 0, 0, "cull"});
  const auto cull_layout = BuildMeshEmulationLayout(cull_fields, corner_input);
  const auto cull_vertex = BuildMeshEmulationVertexShader(cull_layout, 0);
  Check(!SpirvContainsCapability(cull_vertex, 33u), "fallback requires unavailable native cull distance");
  for (const bool visible : {false, true}) {
    std::vector<uint32_t> cull_record(4 + cull_layout.GroupDwords() + 16, 0xcdcdcdcdu);
    cull_record[0] = cull_record[1] = 0;
    cull_record[2] = cull_record[3] = 1;
    cull_record[4] = 3;
    cull_record[5] = 1;
    cull_record[6] = 0x00200400u;
    for (uint32_t v = 0; v < 3; ++v) {
      const auto base = 4 + cull_layout.VerticesOffset() + v * cull_layout.vertex_dwords;
      for (uint32_t c = 0; c < 4; ++c) cull_record[base + c] = std::bit_cast<uint32_t>(positions[v*4+c]);
      cull_record[base + 8] = std::bit_cast<uint32_t>(visible && v == 2 ? 1.f : -1.f);
    }
    gpu.Render({}, cull_layout, cull_vertex, fragment, cull_record,
      {uint8_t(visible ? 255 : 0), 0, 0, 255});
  }
  std::puts("Mesh GPU: cull distances reject only wholly negative triangles");
}
