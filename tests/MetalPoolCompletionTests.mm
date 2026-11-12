// Standalone macOS driver check. Build and run through the campaign heavy.sh lock.
// Pass the MoltenVK library path as the only argument. No game or window is opened.
#define VK_USE_PLATFORM_METAL_EXT
#include <vulkan/vulkan.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <thread>

namespace {

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "MetalPoolCompletionTests: %s\n", message);
		std::exit(1);
	}
}

// vkCmdUpdateBuffer uses the same pool-owned allocator and completion returns as
// tessellation, without requiring a shader pipeline or a game asset.
void CheckPoolRelease(PFN_vkGetDeviceProcAddr get_proc, VkDevice device, VkQueue queue,
                      uint32_t family, const VkPhysicalDeviceMemoryProperties& memory_types,
                      id<MTLDevice> metal) {
#define POOL_PROC(name) const auto name = reinterpret_cast<PFN_##name>(get_proc(device, #name))
	POOL_PROC(vkCreateCommandPool);
	POOL_PROC(vkAllocateCommandBuffers);
	POOL_PROC(vkBeginCommandBuffer);
	POOL_PROC(vkEndCommandBuffer);
	POOL_PROC(vkCmdUpdateBuffer);
	POOL_PROC(vkCmdPipelineBarrier);
	POOL_PROC(vkQueueSubmit);
	POOL_PROC(vkCreateFence);
	POOL_PROC(vkWaitForFences);
	POOL_PROC(vkDestroyFence);
	POOL_PROC(vkResetCommandPool);
	POOL_PROC(vkTrimCommandPool);
	POOL_PROC(vkDestroyCommandPool);
	POOL_PROC(vkCreateBuffer);
	POOL_PROC(vkGetBufferMemoryRequirements);
	POOL_PROC(vkAllocateMemory);
	POOL_PROC(vkBindBufferMemory);
	POOL_PROC(vkMapMemory);
	POOL_PROC(vkUnmapMemory);
	POOL_PROC(vkDestroyBuffer);
	POOL_PROC(vkFreeMemory);
#undef POOL_PROC
	constexpr uint64_t update_bytes = 64 * 1024;
	constexpr uint32_t updates = 16;
	VkBufferCreateInfo buffer_info {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	buffer_info.size = update_bytes * updates;
	buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VkBuffer buffer = VK_NULL_HANDLE;
	Check(vkCreateBuffer(device, &buffer_info, nullptr, &buffer) == VK_SUCCESS, "create transfer buffer");
	VkMemoryRequirements requirements {};
	vkGetBufferMemoryRequirements(device, buffer, &requirements);
	uint32_t memory_type = 0;
	constexpr auto host_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	while (memory_type < memory_types.memoryTypeCount &&
	       (!(requirements.memoryTypeBits & (1u << memory_type)) ||
	        (memory_types.memoryTypes[memory_type].propertyFlags & host_flags) != host_flags)) ++memory_type;
	Check(memory_type < memory_types.memoryTypeCount, "find coherent host memory");
	VkMemoryAllocateInfo allocation {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	allocation.allocationSize = requirements.size;
	allocation.memoryTypeIndex = memory_type;
	VkDeviceMemory memory = VK_NULL_HANDLE;
	Check(vkAllocateMemory(device, &allocation, nullptr, &memory) == VK_SUCCESS, "allocate transfer memory");
	Check(vkBindBufferMemory(device, buffer, memory, 0) == VK_SUCCESS, "bind transfer memory");
	std::array<uint32_t, update_bytes / sizeof(uint32_t)> data {};
	for (unsigned cycle = 0; cycle < 8; ++cycle) {
		VkCommandPoolCreateInfo pool_info {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
		pool_info.queueFamilyIndex = family;
		pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		VkCommandPool pool = VK_NULL_HANDLE;
		Check(vkCreateCommandPool(device, &pool_info, nullptr, &pool) == VK_SUCCESS, "create transfer pool");
		VkCommandBufferAllocateInfo command_info {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
		command_info.commandPool = pool;
		command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		command_info.commandBufferCount = 1;
		VkCommandBuffer command = VK_NULL_HANDLE;
		Check(vkAllocateCommandBuffers(device, &command_info, &command) == VK_SUCCESS, "allocate transfer command");
		VkCommandBufferBeginInfo begin {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
		begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		Check(vkBeginCommandBuffer(command, &begin) == VK_SUCCESS, "begin transfer command");
		for (uint32_t i = 0; i < updates; ++i) {
			data.fill(0xc0ff0000u + cycle * updates + i);
			vkCmdUpdateBuffer(command, buffer, update_bytes * i, update_bytes, data.data());
		}
		VkMemoryBarrier barrier {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
		vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
		                     0, 1, &barrier, 0, nullptr, 0, nullptr);
		Check(vkEndCommandBuffer(command) == VK_SUCCESS, "end transfer command");
		VkSubmitInfo submit {VK_STRUCTURE_TYPE_SUBMIT_INFO};
		submit.commandBufferCount = 1;
		submit.pCommandBuffers = &command;
		Check(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS, "submit transfer command");
		VkFenceCreateInfo fence_info {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
		VkFence fence = VK_NULL_HANDLE;
		Check(vkCreateFence(device, &fence_info, nullptr, &fence) == VK_SUCCESS, "create transfer checkpoint");
		Check(vkQueueSubmit(queue, 0, nullptr, fence) == VK_SUCCESS, "submit transfer checkpoint");
		Check(vkWaitForFences(device, 1, &fence, VK_TRUE, 5'000'000'000) == VK_SUCCESS,
		      "wait for transfer host completion");
		void* mapped = nullptr;
		Check(vkMapMemory(device, memory, 0, buffer_info.size, 0, &mapped) == VK_SUCCESS, "map transfer results");
		const auto* words = static_cast<const uint32_t*>(mapped);
		for (uint32_t i = 0; i < updates; ++i) {
			for (size_t j = 0; j < data.size(); ++j) {
				Check(words[i * data.size() + j] == 0xc0ff0000u + cycle * updates + i,
				      "pool retirement must preserve GPU writes");
			}
		}
		vkUnmapMemory(device, memory);
		const auto retained = metal.currentAllocatedSize;
		Check(vkResetCommandPool(device, pool, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT) == VK_SUCCESS,
		      "reset the transfer pool with release resources");
		const auto reset = metal.currentAllocatedSize;
		vkTrimCommandPool(device, pool, 0);
		const auto trimmed = metal.currentAllocatedSize;
		vkDestroyCommandPool(device, pool, nullptr);
		const auto destroyed = metal.currentAllocatedSize;
		std::printf("pool_cycle=%u driver_bytes=%lu reset_release=%lu trim=%lu destroy=%lu\n",
		            cycle, static_cast<unsigned long>(retained), static_cast<unsigned long>(reset),
		            static_cast<unsigned long>(trimmed), static_cast<unsigned long>(destroyed));
		Check(destroyed + update_bytes * updates <= trimmed,
		      "pool destruction must release the retained temporary Metal buffers");
		vkDestroyFence(device, fence, nullptr);
	}
	vkDestroyBuffer(device, buffer, nullptr);
	vkFreeMemory(device, memory, nullptr);
}

} // namespace

int main(int argc, char** argv) {
	Check(argc == 2, "pass the MoltenVK library path");
	@autoreleasepool {
		void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
		Check(library != nullptr, "load MoltenVK");
		const auto get_instance_proc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
		    dlsym(library, "vkGetInstanceProcAddr"));
		Check(get_instance_proc != nullptr, "load Vulkan dispatch");
		const auto create_instance = reinterpret_cast<PFN_vkCreateInstance>(
		    get_instance_proc(nullptr, "vkCreateInstance"));
		VkApplicationInfo app {VK_STRUCTURE_TYPE_APPLICATION_INFO};
		app.pApplicationName = "MetalPoolCompletionTests";
		app.apiVersion = VK_API_VERSION_1_2;
		VkInstanceCreateInfo instance_info {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
		// This test loads MoltenVK directly, without the Vulkan loader.
		instance_info.pApplicationInfo = &app;
		VkInstance instance = VK_NULL_HANDLE;
		Check(create_instance(&instance_info, nullptr, &instance) == VK_SUCCESS, "create Vulkan instance");
#define INSTANCE_PROC(name) const auto name = reinterpret_cast<PFN_##name>(get_instance_proc(instance, #name))
		INSTANCE_PROC(vkEnumeratePhysicalDevices);
		INSTANCE_PROC(vkGetPhysicalDeviceQueueFamilyProperties);
		INSTANCE_PROC(vkGetPhysicalDeviceMemoryProperties);
		INSTANCE_PROC(vkCreateDevice);
		INSTANCE_PROC(vkGetDeviceProcAddr);
		INSTANCE_PROC(vkDestroyInstance);
		uint32_t physical_count = 1;
		VkPhysicalDevice physical = VK_NULL_HANDLE;
		const auto enumeration = vkEnumeratePhysicalDevices(instance, &physical_count, &physical);
		Check((enumeration == VK_SUCCESS || enumeration == VK_INCOMPLETE) && physical_count > 0,
		      "find a Vulkan device");
		uint32_t family_count = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, nullptr);
		Check(family_count <= 32, "queue family count fits the test");
		VkQueueFamilyProperties families[32] {};
		vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, families);
		uint32_t family = 0;
		while (family < family_count && !(families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT)) ++family;
		Check(family < family_count, "find a graphics queue");
		float priority = 1.0f;
		VkDeviceQueueCreateInfo queue_info {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
		queue_info.queueFamilyIndex = family;
		queue_info.queueCount = 1;
		queue_info.pQueuePriorities = &priority;
		VkPhysicalDeviceTimelineSemaphoreFeatures timeline_feature {
		    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
		timeline_feature.timelineSemaphore = VK_TRUE;
		const char* device_extensions[] = {"VK_KHR_portability_subset", VK_EXT_METAL_OBJECTS_EXTENSION_NAME};
		VkDeviceCreateInfo device_info {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
		device_info.pNext = &timeline_feature;
		device_info.queueCreateInfoCount = 1;
		device_info.pQueueCreateInfos = &queue_info;
		device_info.enabledExtensionCount = 2;
		device_info.ppEnabledExtensionNames = device_extensions;
		VkDevice device = VK_NULL_HANDLE;
		Check(vkCreateDevice(physical, &device_info, nullptr, &device) == VK_SUCCESS, "create Vulkan device");
#define DEVICE_PROC(name) const auto name = reinterpret_cast<PFN_##name>(vkGetDeviceProcAddr(device, #name))
		DEVICE_PROC(vkGetDeviceQueue);
		DEVICE_PROC(vkExportMetalObjectsEXT);
		DEVICE_PROC(vkCreateSemaphore);
		DEVICE_PROC(vkCreateFence);
		DEVICE_PROC(vkQueueSubmit);
		DEVICE_PROC(vkWaitSemaphores);
		DEVICE_PROC(vkWaitForFences);
		DEVICE_PROC(vkDestroyFence);
		DEVICE_PROC(vkDestroySemaphore);
		DEVICE_PROC(vkDestroyDevice);
		Check(vkExportMetalObjectsEXT != nullptr, "load Metal queue export");
		VkQueue queue = VK_NULL_HANDLE;
		vkGetDeviceQueue(device, family, 0, &queue);
		VkExportMetalCommandQueueInfoEXT metal_queue {VK_STRUCTURE_TYPE_EXPORT_METAL_COMMAND_QUEUE_INFO_EXT};
		metal_queue.queue = queue;
		VkExportMetalObjectsInfoEXT metal_objects {VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT};
		metal_objects.pNext = &metal_queue;
		vkExportMetalObjectsEXT(device, &metal_objects);
		id<MTLCommandQueue> native_queue = metal_queue.mtlCommandQueue;
		Check(native_queue != nil, "export Metal command queue");

		VkSemaphoreTypeCreateInfo timeline_type {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
		timeline_type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
		VkSemaphoreCreateInfo semaphore_info {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
		semaphore_info.pNext = &timeline_type;
		VkSemaphore timeline = VK_NULL_HANDLE;
		Check(vkCreateSemaphore(device, &semaphore_info, nullptr, &timeline) == VK_SUCCESS, "create timeline");
		VkFenceCreateInfo fence_info {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
		VkFence fence = VK_NULL_HANDLE;
		Check(vkCreateFence(device, &fence_info, nullptr, &fence) == VK_SUCCESS, "create checkpoint fence");

		// Hold an earlier host completion handler open. GPU events can still advance.
		// The pool checkpoint must stay pending until this handler returns.
		dispatch_semaphore_t entered = dispatch_semaphore_create(0);
		dispatch_semaphore_t release = dispatch_semaphore_create(0);
		std::atomic<bool> handler_finished {false};
		auto* finished = &handler_finished;
		id<MTLCommandBuffer> held = [native_queue commandBuffer];
		[held addCompletedHandler:^(id<MTLCommandBuffer>) {
			dispatch_semaphore_signal(entered);
			dispatch_semaphore_wait(release, DISPATCH_TIME_FOREVER);
			finished->store(true, std::memory_order_release);
		}];
		[held commit];
		Check(dispatch_semaphore_wait(entered, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC)) == 0,
		      "enter the delayed host completion handler");

		uint64_t signal_value = 1;
		VkTimelineSemaphoreSubmitInfo signal {VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
		signal.signalSemaphoreValueCount = 1;
		signal.pSignalSemaphoreValues = &signal_value;
		VkSubmitInfo submit {VK_STRUCTURE_TYPE_SUBMIT_INFO};
		submit.pNext = &signal;
		submit.signalSemaphoreCount = 1;
		submit.pSignalSemaphores = &timeline;
		const auto submit_result = vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
		const auto checkpoint_result = vkQueueSubmit(queue, 0, nullptr, fence);
		VkSemaphoreWaitInfo wait {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
		wait.semaphoreCount = 1;
		wait.pSemaphores = &timeline;
		wait.pValues = &signal_value;
		const auto timeline_result = vkWaitSemaphores(device, &wait, 5'000'000'000);
		id<MTLSharedEvent> gpu_done = [native_queue.device newSharedEvent];
		id<MTLCommandBuffer> observer = [native_queue commandBuffer];
		[observer encodeSignalEvent:gpu_done value:1];
		[observer commit];
		const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
		while (gpu_done.signaledValue < 1 && std::chrono::steady_clock::now() < deadline) {
			std::this_thread::yield();
		}
		const bool gpu_finished = gpu_done.signaledValue >= 1;
		const auto fence_before = vkWaitForFences(device, 1, &fence, VK_TRUE, 20'000'000);
		const bool handler_before = handler_finished.load(std::memory_order_acquire);

		// Always release the handler before checking results, including on a failure.
		dispatch_semaphore_signal(release);
		const auto fence_after = vkWaitForFences(device, 1, &fence, VK_TRUE, 5'000'000'000);
		const bool handler_after = handler_finished.load(std::memory_order_acquire);
		std::printf("timeline=%d checkpoint_before_release=%d checkpoint_after_release=%d handler=%d->%d\n",
		            timeline_result, fence_before, fence_after, handler_before, handler_after);
		Check(submit_result == VK_SUCCESS && checkpoint_result == VK_SUCCESS, "submit the checkpoint");
		Check(timeline_result == VK_SUCCESS && gpu_finished && !handler_before,
		      "GPU timeline must advance while the earlier host handler is held");
		Check(fence_before == VK_TIMEOUT && fence_after == VK_SUCCESS && handler_after,
		      "the checkpoint fence must cover earlier host completion handlers");

		VkPhysicalDeviceMemoryProperties memory_types {};
		vkGetPhysicalDeviceMemoryProperties(physical, &memory_types);
		CheckPoolRelease(vkGetDeviceProcAddr, device, queue, family, memory_types, native_queue.device);
		vkDestroyFence(device, fence, nullptr);
		vkDestroySemaphore(device, timeline, nullptr);
		vkDestroyDevice(device, nullptr);
		vkDestroyInstance(instance, nullptr);
		std::puts("MetalPoolCompletionTests: PASS");
	}
	return 0;
}
