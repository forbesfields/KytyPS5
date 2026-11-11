#include "graphics/host_gpu/renderer/renderContext.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/presentation/videoOut.h"
#include "libs/errno.h"

#include <algorithm>
#include <cinttypes>

namespace Libs::Graphics {

RenderContext::RenderContext(GraphicContext& graphics, const std::filesystem::path& shader_store)
    : m_graphics(graphics), m_render_executor(*this), m_command_scheduler(*this, graphics),
      m_descriptor_heap(graphics, m_command_scheduler.GetMasterSemaphore()),
      m_pipeline_cache(graphics, &m_image_heap, shader_store), m_sampler_cache(graphics),
      m_buffer_cache(graphics, m_command_scheduler, m_page_manager, m_texture_cache),
      m_texture_cache(graphics, m_command_scheduler, m_page_manager, m_buffer_cache),
      m_image_heap(*this, m_page_manager), m_occlusion_queries(graphics, m_command_scheduler),
      m_unified_cache_limit(graphics.GetUnifiedCacheLimit()) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	if (m_unified_cache_limit) {
		vk::PhysicalDeviceDriverProperties driver {};
		vk::PhysicalDeviceProperties2 properties {};
		properties.pNext = &driver;
		graphics.physical_device.getProperties2(&properties);
		m_reclaim_metal_pool = driver.driverID == vk::DriverId::eMoltenvk;
		Log::WriteToConsoleAndLog(fmt::format("Unified GPU cache budget: {} MiB\n",
		                                      *m_unified_cache_limit / (1024 * 1024)));
	}
}

RenderContext::~RenderContext() {
	ShutdownGpu();
	m_command_scheduler.Shutdown();
}

void RenderContext::InitializeGpu(VideoOut::VideoOutDriver* video_out) {
	EXIT_IF(m_gpu != nullptr);
	m_video_out = video_out;
	m_gpu       = std::make_unique<GuestGpu>(*this);
}

void RenderContext::ShutdownGpu() {
	if (m_gpu != nullptr) {
		m_gpu->Shutdown();
		m_gpu.reset();
	}
	if (m_video_out != nullptr) {
		if (m_command_scheduler.Active()) {
			m_command_scheduler.Finish();
		}
		m_command_scheduler.DrainPriorityOperations();
		m_video_out = nullptr;
	}
}

GuestGpu& RenderContext::GetGpu() const {
	EXIT_IF(m_gpu == nullptr);
	return *m_gpu;
}

VideoOut::VideoOutDriver& RenderContext::GetVideoOut() const {
	EXIT_IF(m_video_out == nullptr);
	return *m_video_out;
}

bool RenderContext::HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	// The host reports the faulting byte, not the instruction's access width. Both caches
	// resolve its page; guessing a width can cross the end of a valid guest mapping.
	constexpr uint64_t fault_size = 1;
	if (!IsMapped(fault_vaddr, fault_size)) {
		return false;
	}
	// The access retries once the image write-backs over its page land.
	if (WaitWriteBacks(fault_vaddr, fault_size)) {
		return true;
	}
	if (access == PageFaultAccess::Write) {
		m_buffer_cache.InvalidateMemory(fault_vaddr, fault_size);
		m_texture_cache.InvalidateMemory(fault_vaddr, fault_size);
		m_image_heap.InvalidateMemory(fault_vaddr, fault_size);
	} else {
		m_buffer_cache.ReadMemory(fault_vaddr, fault_size);
	}
	return true;
}

bool RenderContext::WaitWriteBacks(uint64_t vaddr, uint64_t size) {
	const auto tick = m_texture_cache.WriteBackTick(vaddr, size);
	if (tick == 0) {
		return false;
	}
	// The GPU thread can fault inside a draw, so it submits without running deferred operations.
	if (GuestGpu::IsGpuThread()) {
		m_command_scheduler.Wait(tick);
	} else if (tick == m_command_scheduler.CurrentTick()) {
		GetGpu().SendCommandSync([this, tick] { m_command_scheduler.Wait(tick); });
	}
	m_command_scheduler.WaitPriorityOperations(tick);
	return true;
}

bool RenderContext::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	// Host I/O into the range, even one running past its mapping, would fail on its inaccessible
	// pages, and a later write-back would overwrite it.
	(void)WaitWriteBacks(vaddr, size);
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	m_buffer_cache.InvalidateMemory(vaddr, size);
	m_texture_cache.InvalidateMemory(vaddr, size);
	m_image_heap.InvalidateMemory(vaddr, size);
	return true;
}

bool RenderContext::IsMapped(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges.Contains(vaddr, size);
}

void RenderContext::MapMemory(uint64_t vaddr, uint64_t size) {
	std::lock_guard lock(m_mapped_ranges_mutex);
	m_mapped_ranges.Add(vaddr, size);
	m_buffer_cache.QueueBdaSync(vaddr, size);
}

void RenderContext::UnmapMemory(uint64_t vaddr, uint64_t size) {
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	const auto unmap = [this, vaddr, size] {
		// Check cache ownership on the GPU thread. Guest-memory callbacks can still
		// access a range with no cached data, so they must finish before it is unmapped.
		if (m_command_scheduler.Active() &&
		    (m_buffer_cache.IsRegionRegistered(vaddr, size) ||
		     m_texture_cache.IsRegionRegistered(vaddr, size) ||
		     m_command_scheduler.HasPendingPriorityOperations())) {
			const auto tick = m_command_scheduler.CurrentTick();
			m_command_scheduler.Finish();
			m_command_scheduler.WaitPriorityOperations(tick);
		}
		m_buffer_cache.InvalidateMemory(vaddr, size);
		m_texture_cache.UnmapMemory(vaddr, size);
		m_image_heap.UnmapMemory(vaddr, size);
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Subtract(vaddr, size);
	};
	// Shutdown still owns the GPU while queued rendering drains, but its command lane no
	// longer accepts external work. Use the guest GPU's state for the teardown route.
	if (m_gpu == nullptr || m_gpu->IsStopping()) {
		unmap();
		return;
	}
	m_gpu->SendCommandSync(unmap);
}

void RenderContext::PrepareBda() {
	std::shared_lock lock(m_mapped_ranges_mutex);
	m_buffer_cache.TakeBdaSync().ForEach([this](uint64_t begin, uint64_t end) {
		m_mapped_ranges.ForEachInRange(begin, end - begin, [this](uint64_t start, uint64_t finish) {
			m_buffer_cache.SynchronizeBuffersInRange(start, finish - start);
		});
	});
	m_fault_process_pending = true;
}

void RenderContext::PrepareDraw() {
	const bool use_ring = m_command_scheduler.PoolRingEnabled();
	if (!m_reclaim_metal_pool || !m_tessellation_batch.NeedsReclaim(use_ring)) {
		return;
	}
	if (use_ring) {
		// Rotation is asynchronous. Sample driver bytes at the frame boundary,
		// where completed pools have been collected, instead of around each rotation.
		m_command_scheduler.RecreateCommandPool();
		m_tessellation_batch.Reset();
		m_last_pool_reclaim = std::chrono::steady_clock::now();
		return;
	}
	const auto before = m_graphics.GetUnifiedMemoryUsage();
	m_command_scheduler.RecreateCommandPool();
	m_tessellation_batch.Reset();
	m_last_pool_reclaim = std::chrono::steady_clock::now();
	const auto after = m_graphics.GetUnifiedMemoryUsage();
	constexpr uint64_t MiB = 1024 * 1024;
	LOGF("Metal tessellation pool retired: driver=%" PRIu64 " -> %" PRIu64 " MiB\n",
	     before.driver_bytes / MiB, after.driver_bytes / MiB);
}

void RenderContext::RecordTessellationDraw(uint32_t vertices, uint32_t instances, bool indirect) {
	if (m_reclaim_metal_pool) {
		if (indirect) {
			m_tessellation_batch.RecordIndirect();
		} else {
			m_tessellation_batch.Record(vertices, instances);
		}
	}
}

void RenderContext::RunGarbageCollector() {
	m_command_scheduler.AdvanceCommandPoolFrame();
	m_command_scheduler.PopPendingOperations();
	if (m_fault_process_pending) {
		m_fault_process_pending = false;
		m_buffer_cache.ProcessFaultBuffer();
	}
	m_image_heap.RunGarbageCollector();
	m_texture_cache.ProcessDownloadImages();
	if (m_unified_cache_limit) {
		constexpr uint64_t MiB = 1024 * 1024;
		const auto frame = m_budget_log_frame++;
		auto usage = m_graphics.GetUnifiedMemoryUsage();
		GpuCacheBudget budget {usage.driver_bytes, *m_unified_cache_limit,
		                       GraphicContext::GetHostCachePressure()};
		if (m_reclaim_metal_pool &&
		    (m_tessellation_batch.NeedsFrameReclaim(m_command_scheduler.PoolRingEnabled()) ||
		     budget.ShouldReclaimDriverPool(usage.allocator_block_bytes,
		         std::chrono::steady_clock::now() - m_last_pool_reclaim))) {
			const auto before = usage;
			m_command_scheduler.RecreateCommandPool();
			m_tessellation_batch.Reset();
			usage = m_graphics.GetUnifiedMemoryUsage();
			budget.used = usage.driver_bytes;
			budget.host_level = GraphicContext::GetHostCachePressure();
			// Measure the cooldown in wall time so slow frames do not delay critical reclaim.
			m_last_pool_reclaim = std::chrono::steady_clock::now();
			LOGF("Metal command pool retired: driver=%" PRIu64 " -> %" PRIu64
			     " MiB, VMA blocks=%" PRIu64 " -> %" PRIu64 " MiB\n",
			     before.driver_bytes / MiB, usage.driver_bytes / MiB,
			     before.allocator_block_bytes / MiB, usage.allocator_block_bytes / MiB);
		}
		if (frame % 120 == 0) {
			using Common::DebugCounters::Counter;
			using Common::DebugCounters::Get;
			LOGF("Unified GPU cache: used=%" PRIu64 " MiB limit=%" PRIu64
			     " MiB level=%u host=%u texture_evictions=%" PRIu64 " buffer_evictions=%" PRIu64 "\n",
			     budget.used / (1024 * 1024), budget.limit / (1024 * 1024),
			     static_cast<unsigned>(budget.Level(budget.used)), static_cast<unsigned>(budget.host_level),
			     Get(Counter::TextureEvictions), Get(Counter::BufferEvictions));
		}
		m_texture_cache.RunGarbageCollector(&budget);
		m_buffer_cache.RunGarbageCollector(&budget);
	} else {
		m_texture_cache.RunGarbageCollector(nullptr);
		m_buffer_cache.RunGarbageCollector(nullptr);
	}
}

void RenderContext::AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it != m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.push_back({eq, event_id});
}

void RenderContext::DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it == m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.erase(it);
}

void RenderContext::TriggerInterrupt(int event_id, uint32_t context_id) {
	std::vector<InterruptEqRegistration> registrations;
	{
		Common::LockGuard lock(m_interrupt_mutex);
		for (const auto& registration: m_interrupt_eqs) {
			if (registration.event_id == event_id) {
				registrations.push_back(registration);
			}
		}
	}

	for (const auto& registration: registrations) {
		const auto result = LibKernel::EventQueue::KernelTriggerEvent(
		    registration.eq, static_cast<uintptr_t>(registration.event_id),
		    LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS,
		    reinterpret_cast<void*>(static_cast<uintptr_t>(context_id)));
		if (result == LibKernel::KERNEL_ERROR_EBADF || result == LibKernel::KERNEL_ERROR_ENOENT) {
			DeleteInterruptEq(registration.eq, registration.event_id);
			continue;
		}
		EXIT_NOT_IMPLEMENTED(result != OK);
	}
}

} // namespace Libs::Graphics
