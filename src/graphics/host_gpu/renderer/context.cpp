#include "common/assert.h"
#include "common/common.h"
#include "common/debugCounters.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <cstring>
namespace Libs::Graphics {

namespace {

struct PassOptions {
	bool merge;
	bool attachment_ops;
};

const PassOptions& GetPassOptions() {
	static const PassOptions options = [] {
		const auto enabled = [](const char* name) {
			const auto* value = std::getenv(name);
			return value == nullptr || std::strcmp(value, "0") != 0;
		};
		const PassOptions result {enabled("KYTY_PERF_PASS_MERGE"),
		                          enabled("KYTY_PERF_ATTACHMENT_OPS")};
		LOGF("Performance: KYTY_PERF_PASS_MERGE=%u KYTY_PERF_ATTACHMENT_OPS=%u\n",
		     static_cast<unsigned>(result.merge), static_cast<unsigned>(result.attachment_ops));
		return result;
	}();
	return options;
}

// Only the selected aspect's read-only layouts prove it cannot be written in this scope.
// GENERAL and feedback-loop layouts do not provide that proof.
bool ReadOnlyAspect(vk::ImageLayout layout, bool stencil) {
	switch (layout) {
		case vk::ImageLayout::eDepthStencilReadOnlyOptimal: return true;
		case vk::ImageLayout::eDepthReadOnlyOptimal:
		case vk::ImageLayout::eDepthReadOnlyStencilAttachmentOptimal: return !stencil;
		case vk::ImageLayout::eStencilReadOnlyOptimal:
		case vk::ImageLayout::eDepthAttachmentStencilReadOnlyOptimal: return stencil;
		default: return false;
	}
}

RenderState WithoutClearValues(RenderState state) {
	// Color clear values are not used by BeginRendering. Keep all view and layout fields.
	for (auto& color: state.color_attachments) {
		color.clear_value = {};
	}
	state.depth_stencil_attachment.clear_value   = {};
	state.depth_stencil_attachment.depth_clear   = false;
	state.depth_stencil_attachment.stencil_clear = false;
	return state;
}

} // namespace

CommandBuffer::CommandBuffer(CommandScheduler& scheduler)
    : m_context(scheduler.Context()), m_graphics(scheduler.Graphics()) {
	(void)GetPassOptions();
}

void CommandBuffer::Begin() {
	EXIT_IF(m_rendering.Active() || IsInvalid());
	auto buffer = Handle();

	vk::CommandBufferBeginInfo begin_info {};
	begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

	auto result = buffer.begin(&begin_info);

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::End() const {
	EndRendering();
	auto buffer = Handle();

	auto result = buffer.end();

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

void CommandBuffer::BeginRendering(const RenderState& state) const {
	const auto& options = GetPassOptions();
	m_rendering.Begin(Handle(), state, options.merge, options.attachment_ops);
}

void CommandBuffer::EndRendering() const {
	m_rendering.End(Handle());
}

void RenderingScope::Begin(vk::CommandBuffer command, const RenderState& state, bool merge_passes,
                           bool attachment_ops) {
	if (m_active && m_state == state) {
		return;
	}
	if (m_active && merge_passes && WithoutClearValues(m_state) == WithoutClearValues(state)) {
		const auto&         depth_stencil = state.depth_stencil_attachment;
		vk::ClearAttachment clear {};
		if (depth_stencil.has_depth && depth_stencil.depth_clear) {
			clear.aspectMask |= vk::ImageAspectFlagBits::eDepth;
		}
		if (depth_stencil.has_stencil && depth_stencil.stencil_clear) {
			clear.aspectMask |= vk::ImageAspectFlagBits::eStencil;
		}
		if (clear.aspectMask) {
			// Clear at this command position. Vulkan orders attachment clears with rasterization.
			// The matching extent and layer count cover exactly the former load-clear area.
			clear.clearValue.depthStencil = vk::ClearDepthStencilValue {
			    std::bit_cast<float>(depth_stencil.clear_value[0]), depth_stencil.clear_value[1]};
			const vk::ClearRect rect {{{0, 0}, {state.width, state.height}}, 0, state.num_layers};
			command.clearAttachments(1, &clear, 1, &rect);
			Common::DebugCounters::Add(Common::DebugCounters::Counter::InPassClears);
		}
		Common::DebugCounters::Add(Common::DebugCounters::Counter::RenderPassMerges);
		m_state                                        = state;
		m_state.depth_stencil_attachment.depth_clear   = false;
		m_state.depth_stencil_attachment.stencil_clear = false;
		return;
	}
	End(command);

	std::array<vk::RenderingAttachmentInfo, RENDER_COLOR_ATTACHMENTS_MAX> colors {};
	for (uint32_t i = 0; i < state.num_color_attachments; i++) {
		const auto& attachment = state.color_attachments[i];
		colors[i].imageView    = attachment.image_view;
		colors[i].imageLayout  = attachment.image_layout;
		colors[i].loadOp       = vk::AttachmentLoadOp::eLoad;
		colors[i].storeOp      = vk::AttachmentStoreOp::eStore;
	}

	const auto&                 depth_stencil = state.depth_stencil_attachment;
	vk::RenderingAttachmentInfo depth {};
	depth.imageView   = depth_stencil.image_view;
	depth.imageLayout = depth_stencil.image_layout;
	depth.loadOp =
	    depth_stencil.depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	depth.storeOp                       = attachment_ops && !depth_stencil.depth_clear &&
	                                              ReadOnlyAspect(depth_stencil.image_layout, false)
	                                          ? vk::AttachmentStoreOp::eNone
	                                          : vk::AttachmentStoreOp::eStore;
	depth.clearValue.depthStencil.depth = std::bit_cast<float>(depth_stencil.clear_value[0]);

	vk::RenderingAttachmentInfo stencil {};
	stencil.imageView   = depth_stencil.image_view;
	stencil.imageLayout = depth_stencil.image_layout;
	stencil.loadOp =
	    depth_stencil.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	stencil.storeOp = attachment_ops && !depth_stencil.stencil_clear &&
	                          ReadOnlyAspect(depth_stencil.image_layout, true)
	                      ? vk::AttachmentStoreOp::eNone
	                      : vk::AttachmentStoreOp::eStore;
	stencil.clearValue.depthStencil.stencil = depth_stencil.clear_value[1];

	vk::RenderingInfo rendering {};
	rendering.renderArea.extent    = {state.width, state.height};
	rendering.layerCount           = state.num_layers;
	rendering.colorAttachmentCount = state.num_color_attachments;
	rendering.pColorAttachments    = colors.data();
	rendering.pDepthAttachment     = depth_stencil.has_depth ? &depth : nullptr;
	rendering.pStencilAttachment   = depth_stencil.has_stencil ? &stencil : nullptr;
	command.beginRendering(rendering);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::RenderPassStarts);
	if (depth_stencil.has_depth && depth.storeOp == vk::AttachmentStoreOp::eNone) {
		Common::DebugCounters::Add(Common::DebugCounters::Counter::AttachmentStoresSkipped);
	}
	if (depth_stencil.has_stencil && stencil.storeOp == vk::AttachmentStoreOp::eNone) {
		Common::DebugCounters::Add(Common::DebugCounters::Counter::AttachmentStoresSkipped);
	}
	m_state = state;
	// The open pass holds the cleared contents. A later clear must execute again.
	m_state.depth_stencil_attachment.depth_clear   = false;
	m_state.depth_stencil_attachment.stencil_clear = false;
	m_active                                       = true;
}

void RenderingScope::End(vk::CommandBuffer command) {
	if (!m_active) {
		return;
	}
	command.endRendering();
	m_active = false;
}

} // namespace Libs::Graphics
