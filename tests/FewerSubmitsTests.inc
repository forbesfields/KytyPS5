// This fixture uses the production device setup. It does not load a game or a shader.
void CheckFewerSubmits() {
  constexpr const char *name = "FewerSubmits";
  const auto *setting = std::getenv("KYTY_PERF_FEWER_SUBMITS");
  const bool enabled = setting == nullptr || std::strcmp(setting, "0") != 0;
  Config::Load({.printf_direction = Config::LogDirection::Silent});
  Require(name, "video", SDL_InitSubSystem(SDL_INIT_VIDEO), "SDL video init failed");
  WindowContext window;
  window.graphic_ctx.screen_width = 64;
  window.graphic_ctx.screen_height = 64;
  window.window = SDL_CreateWindow(name, 64, 64, SDL_WINDOW_VULKAN | SDL_WINDOW_HIDDEN);
  Require(name, "window", window.window != nullptr, "hidden Vulkan window failed");
  window.CreateVulkan();
  auto &renderer = *window.render_context;
  renderer.InitializeGpu(nullptr);
  auto &gpu = renderer.GetGpu();
  auto &scheduler = renderer.GetCommandScheduler();
  CommandProcessor processor(renderer, 0);
  constexpr uint32_t slices = 17;
  Common::DebugCounters::SetEnabled(true);

  gpu.SendCommandSync([&] {
    processor.BufferInit();
    Buffer output(window.graphic_ctx, scheduler, MemoryUsage::Download, 0,
                  vk::BufferUsageFlagBits::eTransferDst, slices * sizeof(uint32_t));
    const auto fill = [&](uint64_t offset, uint32_t value) {
      const auto command = scheduler.Current().Handle();
      command.fillBuffer(output.Handle(), offset, sizeof(uint32_t), value);
      vk::BufferMemoryBarrier host_read{};
      host_read.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
      host_read.dstAccessMask = vk::AccessFlagBits::eHostRead;
      host_read.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      host_read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      host_read.buffer = output.Handle();
      host_read.offset = offset;
      host_read.size = sizeof(uint32_t);
      command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost,
                              {}, 0, nullptr, 1, &host_read, 0, nullptr);
    };
    const auto first_tick = scheduler.CurrentTick();
    for (uint32_t i = 0; i < slices; ++i) {
      fill(i * sizeof(uint32_t), 0x100u + i);
      processor.BufferFlush(i + 1 < slices);
    }
    const auto submits = scheduler.CurrentTick() - first_tick;
    Require(name, "bounded batching", submits == (enabled ? 5u : 17u),
            "17 slices did not use five bounded submits, or the kill switch changed legacy policy");
    scheduler.Wait(scheduler.CurrentTick() - 1);
    output.Invalidate(0, output.Size());
    for (uint32_t i = 0; i < slices; ++i) {
      uint32_t value = 0;
      std::memcpy(&value, output.Mapped().data() + i * sizeof(uint32_t), sizeof(value));
      Require(name, "GPU output", value == 0x100u + i, "a batched slice lost its GPU write");
    }
    std::printf("[host]    FewerSubmits GPU writes: 17 slices -> %llu submits\n",
                static_cast<unsigned long long>(submits));

    // A guest completion must be submitted even when another slice is ready.
    std::atomic<bool> fence{false};
    const auto fence_tick = scheduler.CurrentTick();
    scheduler.DeferPriorityOperation([&] { fence.store(true); });
    processor.BufferFlush(true);
    Require(name, "guest fence submit", scheduler.CurrentTick() == fence_tick + 1,
            "a ready slice held back a guest completion");
    scheduler.Wait(fence_tick);
    scheduler.WaitPriorityOperations(fence_tick);
    Require(name, "guest fence completion", fence.load(), "the guest fence did not complete");

    // A CPU readback of the current recording submits it before the timeline wait.
    const auto readback_tick = scheduler.CurrentTick();
    fill(0, 0xabcdef01u);
    processor.BufferFlush(true);
    scheduler.Wait(readback_tick);
    Require(name, "readback submit", scheduler.CurrentTick() == readback_tick + 1 &&
                scheduler.IsFree(readback_tick) && !scheduler.HasDeferredSlices(),
            "a readback waited for unsubmitted work or kept stale batch state");
    output.Invalidate(0, sizeof(uint32_t));
    uint32_t value = 0;
    std::memcpy(&value, output.Mapped().data(), sizeof(value));
    Require(name, "readback result", value == 0xabcdef01u, "readback missed the batched write");
  });

  // Seed all slices before the GPU lane runs. This checks the real lane selector and idle flush.
  std::binary_semaphore held{0};
  std::binary_semaphore ready{0};
  gpu.SendCommand([&] { held.release(); ready.acquire(); });
  held.acquire();
  const auto first_tick = scheduler.CurrentTick();
  std::array<uint32_t, 2> nop{KYTY_PM4(2, Pm4::IT_NOP, 0), 0};
  for (uint32_t i = 0; i < slices; ++i) gpu.Submit(nop);
  ready.release();
  gpu.WaitForIdle();
  const auto submits = scheduler.CurrentTick() - first_tick;
  Require(name, "queued slice batching", submits == (enabled ? 5u : 17u) &&
              !scheduler.HasDeferredSlices(),
          "the GPU lane did not batch ready slices or left work held at idle");
  std::printf("[host]    FewerSubmits GPU lane: 17 slices -> %llu submits\n",
              static_cast<unsigned long long>(submits));

  // A wait at the first packet makes no parser progress. It must still submit a prior batch.
  uint32_t wait_label = 0;
  uint64_t deferred_tick = 0;
  gpu.SendCommandSync([&] {
    deferred_tick = scheduler.CurrentTick();
    scheduler.FlushSlice(true);
  });
  const auto label_address = reinterpret_cast<uint64_t>(&wait_label);
  const std::array<uint32_t, 7> wait_packet{
      KYTY_PM4(7, Pm4::IT_WAIT_REG_MEM, 0), 0x13u,
      static_cast<uint32_t>(label_address), static_cast<uint32_t>(label_address >> 32u),
      1u, UINT32_MAX, 0u};
  gpu.Submit(wait_packet);
  const auto master = scheduler.GetMasterSemaphore().Handle();
  vk::SemaphoreWaitInfo wait_info{};
  wait_info.semaphoreCount = 1;
  wait_info.pSemaphores = &master;
  wait_info.pValues = &deferred_tick;
  const bool submitted_before_wait = window.graphic_ctx.device.waitSemaphores(&wait_info,
      2'000'000'000ull) == vk::Result::eSuccess;
  gpu.SendCommandSync([&] { wait_label = 1; });
  gpu.WaitForIdle();
  Require(name, "blocked first packet", submitted_before_wait && !scheduler.HasDeferredSlices(),
          "a blocked first packet left earlier GPU work unsubmitted");

  // Ready work on a second guest queue keeps a submission boundary at each lane switch.
  gpu.SendCommand([&] { held.release(); ready.acquire(); });
  held.acquire();
  const auto switch_tick = scheduler.CurrentTick();
  for (uint32_t i = 0; i < 2; ++i) {
    gpu.Submit(nop);
    gpu.SubmitCompute(0x20u, nop);
  }
  ready.release();
  gpu.WaitForIdle();
  Require(name, "guest queue boundaries", scheduler.CurrentTick() == switch_tick + 4,
          "batching crossed a guest queue switch");

  const auto late_queue_tick = scheduler.CurrentTick();
  gpu.SendCommandSync([&] { scheduler.FlushSlice(true); });
  gpu.SubmitCompute(0x20u, nop);
  gpu.WaitForIdle();
  Require(name, "late guest queue switch", scheduler.CurrentTick() == late_queue_tick + 2,
          "a queue that became ready after deferral shared the previous queue's recording");

  // The callback runner waits on a deliberately blocked submission. Its wait is not a GPU-lane wait.
  vk::SemaphoreTypeCreateInfo type{};
  type.semaphoreType = vk::SemaphoreType::eTimeline;
  vk::SemaphoreCreateInfo create{};
  create.pNext = &type;
  vk::Semaphore gate;
  Require(name, "timeline", window.graphic_ctx.device.createSemaphore(&create, nullptr, &gate) ==
              vk::Result::eSuccess, "test timeline creation failed");
  const auto before_background = CommandScheduler::GetGpuThreadWaitStats();
  gpu.SendCommandSync([&] {
    SubmitInfo submit;
    submit.AddWait(gate, 1);
    scheduler.DeferPriorityOperation([] {});
    scheduler.Flush(submit);
    const auto old_completion_tick = scheduler.CurrentTick();
    scheduler.FlushSlice(true);
    Require(name, "submitted completion", scheduler.CurrentTick() ==
                old_completion_tick + (enabled ? 0u : 1u),
            "an already submitted completion prevented batching a new slice");
    scheduler.FlushSlice(false);
  });
  const auto during_background = CommandScheduler::GetGpuThreadWaitStats();
  Require(name, "background metric", during_background.count == before_background.count &&
              during_background.nanoseconds == before_background.nanoseconds,
          "the callback runner changed the GPU-thread wait metric");
  vk::SemaphoreSignalInfo signal{};
  signal.semaphore = gate;
  signal.value = 1;
  Require(name, "release timeline", window.graphic_ctx.device.signalSemaphore(&signal) ==
              vk::Result::eSuccess, "test timeline signal failed");
  gpu.SendCommandSync([&] { scheduler.DrainPriorityOperations(); });
  const auto after_background = CommandScheduler::GetGpuThreadWaitStats();
  Require(name, "completed background metric", after_background.count == before_background.count &&
              after_background.nanoseconds == before_background.nanoseconds,
          "a completed callback-runner wait changed the GPU-thread wait metric");

  uint64_t blocked_tick = 0;
  gpu.SendCommandSync([&] {
    blocked_tick = scheduler.CurrentTick();
    SubmitInfo submit;
    submit.AddWait(gate, 2);
    scheduler.Flush(submit);
  });
  std::jthread signaler([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    signal.value = 2;
    Require(name, "release GPU wait", window.graphic_ctx.device.signalSemaphore(&signal) ==
                vk::Result::eSuccess, "GPU-wait timeline signal failed");
  });
  gpu.SendCommandSync([&] { scheduler.Wait(blocked_tick); });
  signaler.join();
  const auto after_gpu_wait = CommandScheduler::GetGpuThreadWaitStats();
  Require(name, "GPU thread metric", after_gpu_wait.count == after_background.count + 1 &&
              after_gpu_wait.nanoseconds > after_background.nanoseconds,
          "a GPU-thread timeline wait was not counted");
  window.graphic_ctx.device.destroySemaphore(gate, nullptr);
  renderer.ShutdownGpu();
  Common::DebugCounters::SetEnabled(false);
  std::printf("[host]    %-32s ok (policy %s)\n", name, enabled ? "on" : "legacy");
}
