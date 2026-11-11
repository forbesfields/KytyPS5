#include "common/debugCounters.h"
#include "graphics/host_gpu/renderer/render.h"

#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace {
using namespace Libs::Graphics;
using Counter = Common::DebugCounters::Counter;

#define CHECK(condition)                                                                           \
	do {                                                                                           \
		if (!(condition)) {                                                                        \
			std::fprintf(stderr, "RenderingScopeTests:%d: %s\n", __LINE__, #condition);            \
			std::abort();                                                                          \
		}                                                                                          \
	} while (false)

std::string               events;
VkRenderingAttachmentInfo recorded_depth {};
VkRenderingAttachmentInfo recorded_stencil {};
VkClearAttachment         recorded_clear {};
VkClearRect               recorded_rect {};
uint32_t                  begins = 0;

VKAPI_ATTR void VKAPI_CALL RecordBegin(VkCommandBuffer, const VkRenderingInfo* info) {
	events += 'B';
	++begins;
	recorded_depth =
	    info->pDepthAttachment ? *info->pDepthAttachment : VkRenderingAttachmentInfo {};
	recorded_stencil =
	    info->pStencilAttachment ? *info->pStencilAttachment : VkRenderingAttachmentInfo {};
	for (uint32_t i = 0; i < info->colorAttachmentCount; ++i) {
		CHECK(info->pColorAttachments[i].loadOp == VK_ATTACHMENT_LOAD_OP_LOAD);
		CHECK(info->pColorAttachments[i].storeOp == VK_ATTACHMENT_STORE_OP_STORE);
	}
}
VKAPI_ATTR void VKAPI_CALL RecordEnd(VkCommandBuffer) {
	events += 'E';
}
VKAPI_ATTR void VKAPI_CALL RecordClear(VkCommandBuffer, uint32_t count,
                                       const VkClearAttachment* clears, uint32_t rect_count,
                                       const VkClearRect* rects) {
	CHECK(count == 1 && rect_count == 1);
	events += 'C';
	recorded_clear = clears[0];
	recorded_rect  = rects[0];
}

RenderState State() {
	RenderState state {};
	state.width                             = 4;
	state.height                            = 4;
	state.num_layers                        = 2;
	state.num_color_attachments             = 1;
	state.color_attachments[0].image_view   = reinterpret_cast<VkImageView>(uintptr_t {1});
	state.color_attachments[0].image_layout = vk::ImageLayout::eColorAttachmentOptimal;
	auto& depth                             = state.depth_stencil_attachment;
	depth.image_view                        = reinterpret_cast<VkImageView>(uintptr_t {2});
	depth.image_layout                      = vk::ImageLayout::eDepthStencilAttachmentOptimal;
	depth.has_depth                         = true;
	depth.has_stencil                       = true;
	depth.clear_value                       = {std::bit_cast<uint32_t>(0.75f), 91};
	return state;
}

void CheckCommandOrder(bool merge) {
	RenderingScope          scope;
	auto                    state   = State();
	const vk::CommandBuffer command = reinterpret_cast<VkCommandBuffer>(uintptr_t {1});
	events.clear();
	begins = 0;
	scope.Begin(command, state, merge, false);
	events += 'D'; // A draw recorded by the caller.
	state.depth_stencil_attachment.depth_clear    = true;
	state.depth_stencil_attachment.clear_value[0] = std::bit_cast<uint32_t>(0.25f);
	scope.Begin(command, state, merge, false);
	if (merge) {
		CHECK(events == "BDC");
		CHECK(recorded_clear.aspectMask == VK_IMAGE_ASPECT_DEPTH_BIT);
		CHECK(recorded_clear.clearValue.depthStencil.depth == 0.25f);
		CHECK(recorded_rect.rect.offset.x == 0 && recorded_rect.rect.offset.y == 0);
		CHECK(recorded_rect.rect.extent.width == 4 && recorded_rect.rect.extent.height == 4);
		CHECK(recorded_rect.baseArrayLayer == 0 && recorded_rect.layerCount == 2);
	} else {
		CHECK(events == "BDEB");
		CHECK(recorded_depth.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR);
	}
	// Repeated equal clears must execute again, even when values match.
	scope.Begin(command, state, merge, false);
	events += 'D';
	state.depth_stencil_attachment.depth_clear    = false;
	state.depth_stencil_attachment.stencil_clear  = true;
	state.depth_stencil_attachment.clear_value[1] = 17;
	scope.Begin(command, state, merge, false);
	if (merge) {
		CHECK(recorded_clear.aspectMask == VK_IMAGE_ASPECT_STENCIL_BIT);
		CHECK(recorded_clear.clearValue.depthStencil.stencil == 17);
	}
	state.depth_stencil_attachment.depth_clear = true;
	scope.Begin(command, state, merge, false);
	if (merge)
		CHECK(recorded_clear.aspectMask ==
		      (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT));
	state.depth_stencil_attachment.depth_clear   = false;
	state.depth_stencil_attachment.stencil_clear = false;
	const auto before_value_change               = begins;
	state.depth_stencil_attachment.clear_value   = {std::bit_cast<uint32_t>(0.5f), 23};
	scope.Begin(command, state, merge, false);
	CHECK(begins == before_value_change + (merge ? 0u : 1u));
	// Equal state adds no command, including after an ignored value change.
	const auto after_value_change = events;
	scope.Begin(command, state, merge, false);
	CHECK(events == after_value_change);
	scope.End(command);
	scope.End(command);
	if (merge) CHECK(events == "BDCCDCCE");
	CHECK(begins == (merge ? 1u : 6u));
	CHECK(!scope.Active());
	std::printf("Recorded command sequence: merge=%u render_pass_starts=%u\n",
	            static_cast<unsigned>(merge), begins);
}

void CheckBoundaries() {
	const vk::CommandBuffer command  = reinterpret_cast<VkCommandBuffer>(uintptr_t {1});
	const auto              original = State();
	// All attachment identity, layout and render-area changes require a new scope.
	for (int change = 0; change < 9; ++change) {
		RenderingScope scope;
		auto           next = original;
		switch (change) {
			case 0: ++next.width; break;
			case 1: ++next.height; break;
			case 2: ++next.num_layers; break;
			case 3: next.num_color_attachments = 0; break;
			case 4: next.color_attachments[0].image_view = nullptr; break;
			case 5: next.color_attachments[0].image_layout = vk::ImageLayout::eGeneral; break;
			case 6: next.depth_stencil_attachment.image_view = nullptr; break;
			case 7: next.depth_stencil_attachment.image_layout = vk::ImageLayout::eGeneral; break;
			case 8: next.depth_stencil_attachment.has_stencil = false; break;
		}
		events.clear();
		scope.Begin(command, original, true, true);
		scope.Begin(command, next, true, true);
		scope.End(command);
		CHECK(events == "BEBE");
	}
	RenderingScope scope;
	events.clear();
	scope.Begin(command, original, true, true);
	events += 'D';
	scope.End(command);
	events += 'T'; // A copy, dispatch or external barrier stays between the scopes.
	scope.Begin(command, original, true, true);
	events += 'D';
	scope.End(command);
	CHECK(events == "BDETBDE");
}

void CheckStoreOps() {
	const vk::CommandBuffer command = reinterpret_cast<VkCommandBuffer>(uintptr_t {1});
	struct Case {
		vk::ImageLayout layout;
		bool            depth_read_only;
		bool            stencil_read_only;
	};
	const std::array cases {
	    Case {vk::ImageLayout::eDepthStencilReadOnlyOptimal, true, true},
	    Case {vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal, true, false},
	    Case {vk::ImageLayout::eDepthAttachmentStencilReadOnlyOptimal, false, true},
	    Case {vk::ImageLayout::eDepthReadOnlyOptimal, true, false},
	    Case {vk::ImageLayout::eStencilReadOnlyOptimal, false, true},
	    Case {vk::ImageLayout::eDepthStencilAttachmentOptimal, false, false},
	    Case {vk::ImageLayout::eGeneral, false, false},
	    Case {vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT, false, false},
	};
	for (bool optimize: {false, true}) {
		for (const auto& test: cases) {
			RenderingScope scope;
			auto           state                        = State();
			state.depth_stencil_attachment.image_layout = test.layout;
			scope.Begin(command, state, true, optimize);
			CHECK(recorded_depth.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD);
			CHECK(recorded_stencil.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD);
			CHECK(recorded_depth.storeOp == (optimize && test.depth_read_only
			                                     ? VK_ATTACHMENT_STORE_OP_NONE
												 : VK_ATTACHMENT_STORE_OP_STORE));
			CHECK(recorded_stencil.storeOp == (optimize && test.stencil_read_only
			                                       ? VK_ATTACHMENT_STORE_OP_NONE
												   : VK_ATTACHMENT_STORE_OP_STORE));
			scope.End(command);
		}
		RenderingScope scope;
		auto           state                         = State();
		state.depth_stencil_attachment.depth_clear   = true;
		state.depth_stencil_attachment.stencil_clear = true;
		scope.Begin(command, state, true, optimize);
		CHECK(recorded_depth.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR);
		CHECK(recorded_stencil.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR);
		CHECK(recorded_depth.storeOp == VK_ATTACHMENT_STORE_OP_STORE);
		CHECK(recorded_stencil.storeOp == VK_ATTACHMENT_STORE_OP_STORE);
		scope.End(command);
	}
}

void HostTests() {
	VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdBeginRendering   = RecordBegin;
	VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdEndRendering     = RecordEnd;
	VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdClearAttachments = RecordClear;
	Common::DebugCounters::SetEnabled(true);
	CheckCommandOrder(true);
	CHECK(Common::DebugCounters::Get(Counter::RenderPassStarts) == 1);
	CHECK(Common::DebugCounters::Get(Counter::RenderPassMerges) == 5);
	CHECK(Common::DebugCounters::Get(Counter::InPassClears) == 4);
	CheckCommandOrder(false);
	CHECK(Common::DebugCounters::Get(Counter::RenderPassStarts) == 7);
	CheckBoundaries();
	CheckStoreOps();
	CHECK(Common::DebugCounters::Get(Counter::AttachmentStoresSkipped) == 6);
	Common::DebugCounters::SetEnabled(false);
	std::puts("Rendering scope command-order tests passed (merge on/off, clears, boundaries, store "
	          "ops, counters)");
}

// The GPU fixture needs only Vulkan 1.3 dynamic rendering. It does not require the
// emulator shader harness's optional features, which MoltenVK does not support.
struct ImageAllocation {
	vk::Image        image;
	vk::DeviceMemory memory;
	vk::ImageView    view;
};

uint32_t MemoryType(vk::PhysicalDevice physical, uint32_t mask, vk::MemoryPropertyFlags required) {
	const auto properties = physical.getMemoryProperties();
	for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
		if ((mask & (1u << i)) && (properties.memoryTypes[i].propertyFlags & required) == required)
			return i;
	}
	CHECK(false);
	return 0;
}

ImageAllocation MakeImage(vk::Device device, vk::PhysicalDevice physical, vk::Format format,
                          vk::ImageUsageFlags usage, vk::ImageAspectFlags aspects) {
	vk::ImageCreateInfo info {};
	info.imageType   = vk::ImageType::e2D;
	info.format      = format;
	info.extent      = {4, 4, 1};
	info.mipLevels   = 1;
	info.arrayLayers = 2;
	info.samples     = vk::SampleCountFlagBits::e1;
	info.tiling      = vk::ImageTiling::eOptimal;
	info.usage =
	    usage | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
	ImageAllocation result {};
	CHECK(device.createImage(&info, nullptr, &result.image) == vk::Result::eSuccess);
	const auto             requirements = device.getImageMemoryRequirements(result.image);
	vk::MemoryAllocateInfo allocation {};
	allocation.allocationSize  = requirements.size;
	allocation.memoryTypeIndex = MemoryType(physical, requirements.memoryTypeBits, {});
	CHECK(device.allocateMemory(&allocation, nullptr, &result.memory) == vk::Result::eSuccess);
	CHECK(device.bindImageMemory(result.image, result.memory, 0) == vk::Result::eSuccess);
	vk::ImageViewCreateInfo view {};
	view.image            = result.image;
	view.viewType         = vk::ImageViewType::e2DArray;
	view.format           = format;
	view.subresourceRange = {aspects, 0, 1, 0, 2};
	CHECK(device.createImageView(&view, nullptr, &result.view) == vk::Result::eSuccess);
	return result;
}

void Transition(vk::CommandBuffer command, vk::Image image, vk::ImageAspectFlags aspects,
                vk::ImageLayout before, vk::ImageLayout after) {
	vk::ImageMemoryBarrier barrier {};
	barrier.srcAccessMask =
	    before == vk::ImageLayout::eUndefined
	        ? vk::AccessFlags {}
	        : vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	barrier.oldLayout     = before;
	barrier.newLayout     = after;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image               = image;
	barrier.subresourceRange    = {aspects, 0, 1, 0, 2};
	command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                        vk::PipelineStageFlagBits::eAllCommands, {}, 0, nullptr, 0, nullptr, 1,
	                        &barrier);
}

int GpuTests() {
	const auto* library = std::getenv("KYTY_TEST_VULKAN_LIBRARY");
	if (library == nullptr) {
		std::puts("SKIP: set KYTY_TEST_VULKAN_LIBRARY for the rendering scope GPU test");
		return 77;
	}
	vk::detail::DynamicLoader loader(library);
	const auto get_proc = loader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
	CHECK(get_proc != nullptr);
	VULKAN_HPP_DEFAULT_DISPATCHER.init(get_proc);
	vk::ApplicationInfo application {};
	application.pApplicationName = "RenderingScopeTests";
	application.apiVersion       = VK_API_VERSION_1_3;
	std::vector<const char*> extensions;
	for (const auto& extension: vk::enumerateInstanceExtensionProperties().value) {
		if (std::strcmp(extension.extensionName, "VK_KHR_portability_enumeration") == 0)
			extensions.push_back("VK_KHR_portability_enumeration");
	}
	vk::InstanceCreateInfo instance_info {};
	instance_info.pApplicationInfo        = &application;
	instance_info.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
	instance_info.ppEnabledExtensionNames = extensions.data();
	if (!extensions.empty())
		instance_info.flags = vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
	vk::Instance instance;
	CHECK(vk::createInstance(&instance_info, nullptr, &instance) == vk::Result::eSuccess);
	VULKAN_HPP_DEFAULT_DISPATCHER.init(instance);
	const auto devices = instance.enumeratePhysicalDevices();
	CHECK(devices.result == vk::Result::eSuccess && !devices.value.empty());
	const auto physical = devices.value.front();
	uint32_t   family   = UINT32_MAX;
	const auto families = physical.getQueueFamilyProperties();
	for (uint32_t i = 0; i < families.size(); ++i) {
		if (families[i].queueFlags & vk::QueueFlagBits::eGraphics) {
			family = i;
			break;
		}
	}
	CHECK(family != UINT32_MAX);
	const float               priority = 1.0f;
	vk::DeviceQueueCreateInfo queue_info {};
	queue_info.queueFamilyIndex = family;
	queue_info.queueCount       = 1;
	queue_info.pQueuePriorities = &priority;
	vk::PhysicalDeviceVulkan13Features features13 {};
	vk::PhysicalDeviceFeatures2        features {};
	features.pNext = &features13;
	physical.getFeatures2(&features);
	CHECK(features13.dynamicRendering);
	features13                  = {};
	features13.dynamicRendering = true;
	extensions.clear();
	for (const auto& extension: physical.enumerateDeviceExtensionProperties().value) {
		if (std::strcmp(extension.extensionName, "VK_KHR_portability_subset") == 0)
			extensions.push_back("VK_KHR_portability_subset");
	}
	vk::DeviceCreateInfo device_info {};
	device_info.pNext                   = &features13;
	device_info.queueCreateInfoCount    = 1;
	device_info.pQueueCreateInfos       = &queue_info;
	device_info.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
	device_info.ppEnabledExtensionNames = extensions.data();
	vk::Device device;
	CHECK(physical.createDevice(&device_info, nullptr, &device) == vk::Result::eSuccess);
	VULKAN_HPP_DEFAULT_DISPATCHER.init(device);
	const auto queue = device.getQueue(family, 0);
	std::printf("GPU: %s\n", physical.getProperties().deviceName.data());

	const auto both = vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
	const auto color =
	    MakeImage(device, physical, vk::Format::eR8G8B8A8Unorm,
		          vk::ImageUsageFlagBits::eColorAttachment, vk::ImageAspectFlagBits::eColor);
	const auto           depth = MakeImage(device, physical, vk::Format::eD32SfloatS8Uint,
	                                       vk::ImageUsageFlagBits::eDepthStencilAttachment, both);
	vk::BufferCreateInfo buffer_info {};
	buffer_info.size  = 512;
	buffer_info.usage = vk::BufferUsageFlagBits::eTransferDst;
	vk::Buffer readback;
	CHECK(device.createBuffer(&buffer_info, nullptr, &readback) == vk::Result::eSuccess);
	const auto             requirements = device.getBufferMemoryRequirements(readback);
	vk::MemoryAllocateInfo allocation {};
	allocation.allocationSize  = requirements.size;
	allocation.memoryTypeIndex = MemoryType(physical, requirements.memoryTypeBits,
	                                        vk::MemoryPropertyFlagBits::eHostVisible |
	                                            vk::MemoryPropertyFlagBits::eHostCoherent);
	vk::DeviceMemory readback_memory;
	CHECK(device.allocateMemory(&allocation, nullptr, &readback_memory) == vk::Result::eSuccess);
	CHECK(device.bindBufferMemory(readback, readback_memory, 0) == vk::Result::eSuccess);
	vk::CommandPoolCreateInfo pool_info {};
	pool_info.queueFamilyIndex = family;
	vk::CommandPool pool;
	CHECK(device.createCommandPool(&pool_info, nullptr, &pool) == vk::Result::eSuccess);
	vk::CommandBufferAllocateInfo command_info {};
	command_info.commandPool        = pool;
	command_info.level              = vk::CommandBufferLevel::ePrimary;
	command_info.commandBufferCount = 1;
	vk::CommandBuffer command;
	CHECK(device.allocateCommandBuffers(&command_info, &command) == vk::Result::eSuccess);

	for (uint32_t trial = 0; trial < 12; ++trial) {
		const auto                 mode           = trial % 4;
		const auto                 pattern        = trial / 4;
		const bool                 merge          = (mode & 1u) != 0;
		const bool                 attachment_ops = (mode & 2u) != 0;
		vk::CommandBufferBeginInfo begin {};
		CHECK(command.begin(&begin) == vk::Result::eSuccess);
		const auto old_layout =
		    trial != 0 ? vk::ImageLayout::eTransferSrcOptimal : vk::ImageLayout::eUndefined;
		Transition(command, color.image, vk::ImageAspectFlagBits::eColor, old_layout,
		           vk::ImageLayout::eTransferDstOptimal);
		Transition(command, depth.image, both, old_layout, vk::ImageLayout::eTransferDstOptimal);
		vk::ClearColorValue color_value {};
		color_value.float32 = std::array<float, 4> {0.2f, 0.4f, 0.6f, 1.0f};
		const vk::ImageSubresourceRange color_range {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 2};
		const vk::ImageSubresourceRange depth_range {both, 0, 1, 0, 2};
		command.clearColorImage(color.image, vk::ImageLayout::eTransferDstOptimal, &color_value, 1,
		                        &color_range);
		const vk::ClearDepthStencilValue initial {0.75f, 91};
		command.clearDepthStencilImage(depth.image, vk::ImageLayout::eTransferDstOptimal, &initial,
		                               1, &depth_range);
		Transition(command, color.image, vk::ImageAspectFlagBits::eColor,
		           vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eColorAttachmentOptimal);
		Transition(command, depth.image, both, vk::ImageLayout::eTransferDstOptimal,
		           vk::ImageLayout::eDepthStencilAttachmentOptimal);
		auto state                                = State();
		state.color_attachments[0].image_view     = color.view;
		state.depth_stencil_attachment.image_view = depth.view;
		RenderingScope scope;
		scope.Begin(command, state, merge, attachment_ops);
		// Test both clear orders. The second clear must preserve the first clear's aspect.
		state.depth_stencil_attachment.clear_value   = {std::bit_cast<uint32_t>(0.25f), 17};
		state.depth_stencil_attachment.depth_clear   = pattern != 1;
		state.depth_stencil_attachment.stencil_clear = pattern == 1;
		scope.Begin(command, state, merge, attachment_ops);
		state.depth_stencil_attachment.depth_clear   = pattern == 1;
		state.depth_stencil_attachment.stencil_clear = pattern != 1;
		scope.Begin(command, state, merge, attachment_ops);
		if (pattern == 2) {
			state.depth_stencil_attachment.depth_clear    = true;
			state.depth_stencil_attachment.clear_value[0] = std::bit_cast<uint32_t>(0.5f);
			scope.Begin(command, state, merge, attachment_ops);
		}
		scope.End(command);
		Transition(command, depth.image, both, vk::ImageLayout::eDepthStencilAttachmentOptimal,
		           vk::ImageLayout::eDepthStencilReadOnlyOptimal);
		state.depth_stencil_attachment.depth_clear   = false;
		state.depth_stencil_attachment.stencil_clear = false;
		state.depth_stencil_attachment.image_layout = vk::ImageLayout::eDepthStencilReadOnlyOptimal;
		scope.Begin(command, state, merge, attachment_ops);
		scope.End(command);
		Transition(command, color.image, vk::ImageAspectFlagBits::eColor,
		           vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::eTransferSrcOptimal);
		Transition(command, depth.image, both, vk::ImageLayout::eDepthStencilReadOnlyOptimal,
		           vk::ImageLayout::eTransferSrcOptimal);
		for (const auto aspect: {vk::ImageAspectFlagBits::eColor, vk::ImageAspectFlagBits::eDepth,
		                         vk::ImageAspectFlagBits::eStencil}) {
			vk::BufferImageCopy copy {};
			copy.bufferOffset     = aspect == vk::ImageAspectFlagBits::eColor   ? 0
			                        : aspect == vk::ImageAspectFlagBits::eDepth ? 128
			                                                                    : 256;
			copy.imageSubresource = {aspect, 0, 0, 2};
			copy.imageExtent      = {4, 4, 1};
			command.copyImageToBuffer(aspect == vk::ImageAspectFlagBits::eColor ? color.image
			                                                                    : depth.image,
			                          vk::ImageLayout::eTransferSrcOptimal, readback, 1, &copy);
		}
		vk::MemoryBarrier host {};
		host.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		host.dstAccessMask = vk::AccessFlagBits::eHostRead;
		command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                        vk::PipelineStageFlagBits::eHost, {}, 1, &host, 0, nullptr, 0,
		                        nullptr);
		CHECK(command.end() == vk::Result::eSuccess);
		vk::SubmitInfo submit {};
		submit.commandBufferCount = 1;
		submit.pCommandBuffers    = &command;
		CHECK(queue.submit(1, &submit, nullptr) == vk::Result::eSuccess);
		CHECK(queue.waitIdle() == vk::Result::eSuccess);
		void* mapped = nullptr;
		CHECK(device.mapMemory(readback_memory, 0, 512, {}, &mapped) == vk::Result::eSuccess);
		const auto* bytes = static_cast<const uint8_t*>(mapped);
		for (uint32_t pixel = 0; pixel < 32; ++pixel) {
			CHECK(bytes[pixel * 4] == 51 && bytes[pixel * 4 + 1] == 102 &&
			      bytes[pixel * 4 + 2] == 153 && bytes[pixel * 4 + 3] == 255);
			float value = 0;
			std::memcpy(&value, bytes + 128 + pixel * 4, sizeof(value));
			CHECK(value == (pattern == 2 ? 0.5f : 0.25f) && bytes[256 + pixel] == 17);
		}
		device.unmapMemory(readback_memory);
		std::printf("Rendering scope GPU pixels passed (merge=%u attachment_ops=%u pattern=%u, "
		            "32 pixels, color/depth/stencil)\n",
		            static_cast<unsigned>(merge), static_cast<unsigned>(attachment_ops), pattern);
		CHECK(device.resetCommandPool(pool) == vk::Result::eSuccess);
	}
	device.destroyCommandPool(pool);
	device.destroyBuffer(readback);
	device.freeMemory(readback_memory);
	for (const auto& image: {color, depth}) {
		device.destroyImageView(image.view);
		device.destroyImage(image.image);
		device.freeMemory(image.memory);
	}
	device.destroy();
	instance.destroy();
	return 0;
}

} // namespace

int main(int argc, char** argv) {
	if (argc == 2 && std::strcmp(argv[1], "--gpu") == 0) return GpuTests();
	HostTests();
}
